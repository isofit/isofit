# %%
import netCDF4 as nc
import pandas as pd
import os
import numpy as np
import rasterio
from osgeo import gdal, osr
from os.path import join, splitext, basename

def nc2envi(infile: str, fwhmfile: str, outpath: str=os.getcwd(), rtm='srtmnet') -> None:
    """
    Convert an OCI L1B (V2) netcdf file to three ENVI files compliant with ISOFIT
    desired format (and MODTRAN conventions). Can be used for either sRTMnet or 
    6S conversions. 
    
    Requirements:
        infile:         input L1B OCI file to convert 
        fwhmfile:       anc file containing FWHM and wavelength information
        outpath:        directory to put output files
        rtm:            which RTM will be used in ISOFIT atm correction. Options
                        = 'srtmnet' (default), '6s'

    Returns:
        Creates 3 ENVI formatted files containing radiance (rdn), geolocation 
        (loc), and angular (obs) information for the corresponding netcdf. Files
        are place in the outpath directory
    """
    # Open the netcdf and read in relevant information
    try:
        nc_whole = nc.Dataset(infile)
        nc_geo = nc_whole['geolocation_data']
        nc_sbp = nc_whole['sensor_band_parameters']
        nc_obs = nc_whole['observation_data']
        nc_time = np.asarray(nc_whole['scan_line_attributes']['time'][:])
        # Wavelengths and fwhm for metadata, which is read by ISOFIT
        meta_wvs, fwhm = np.loadtxt(fwhmfile, usecols=(1, 2), unpack=True, dtype='str')
    except:
        raise Exception("Problem importing one or more groups from input file.",
                        "Please ensure file is in OCI L1B format")
    
    # Define rtm-specific parameters for rfl -> rdn conversion
    if rtm == 'srtmnet':
        # Exclude 314:377 bc srtmnet only reliable down to 380
        blue_wvs = np.asarray(nc_sbp['blue_wavelength'][28:-3])                 # [:-3] removes overlap with FPAs
        blue_f0= np.asarray(nc_sbp['blue_solar_irradiance'][28:-3] / 10)
        blue_rfl = np.asarray(nc_obs['rhot_blue'][28:-3])
    elif rtm == '6s':
        # 6S can handle down to 314
        blue_wvs = np.asarray(nc_sbp['blue_wavelength'][:-3])
        blue_f0= np.asarray(nc_sbp['blue_solar_irradiance'][:-3] / 10)
        blue_rfl = np.asarray(nc_obs['rhot_blue'][:-3])  
    
    # Hardcoding in SWIR bands bc  939 nm is masked in the nc file, 
    # also dropping high gain bands for now
    swir_wvs = np.asarray([939.713,1038.317,1248.55,1378.169,1618.034,
                           2130.593,2258.429])
    swir_f0 = np.asarray([81.9699, 67.0309, 44.4994, 35.5744, 23.4977, 9.1108, 
                          7.3956])
    swir_all = np.asarray(nc_obs['rhot_SWIR'][:])
    swir_rfl = np.delete(swir_all, [2, 5], axis=0)                              # remove high gain band measurements  
    
    red_wvs = np.asarray(nc_sbp['red_wavelength'][:])
    red_f0 = np.asarray(nc_sbp['red_solar_irradiance'][:] / 10)
    red_rfl = np.asarray(nc_obs['rhot_red'][:])
    
    # Make each parameter one array
    wvs = np.concatenate((blue_wvs, red_wvs, swir_wvs))
    f0s = np.concatenate((blue_f0, red_f0, swir_f0))
    rfls = np.concatenate((blue_rfl, red_rfl, swir_rfl))

    # Define other necessary parameters for ENVI conversion 
    rows, cols = nc_geo['latitude'].shape[0], nc_geo['latitude'].shape[1]
    lat, lon = np.asarray(nc_geo['latitude'][:]), np.asarray(nc_geo['longitude'][:])
    alt = 676.5e3
    sza = np.asarray(nc_geo['solar_zenith'][:])                                 # 0 - 90 from zenith
    saa = np.asarray(nc_geo['solar_azimuth'][:] + 180)                          # 0 - 360 clockwise from north
    vza = np.asarray(nc_geo['sensor_zenith'][:])                                # 0 - 90 from zenith
    vaa = np.asarray(nc_geo['sensor_azimuth'][:] + 180)                         # 0 - 360 clockwise from north

    height = np.asarray(nc_geo['height'][:])
    phang = calc_phase_angle(sza, saa, vza, vaa)
    utctime = calc_time(nc_time, rows, cols)

    # direct geometric distance b/w target and sensor
    pathlength = np.asarray(alt - height)

    # Get flight line id and outfile names
    # Fname must not have "." before format indicator, so replace with "_"
    fid = splitext(basename(infile))[0][0:8] + "_" + splitext(basename(infile))[0][9:24]
    loc_outfile = join(outpath, (fid+"_loc"))
    obs_outfile = join(outpath, (fid+"_obs"))
    rdn_outfile = join(outpath, (fid+"_rdn"))

    driver = gdal.GetDriverByName('ENVI')
    loc_envi = driver.Create(loc_outfile, xsize = cols, ysize = rows, bands = 3, 
                             eType = gdal.GDT_Float64, options = ['INTERLEAVE=BIL'])
    
    # Can add an 11th E-S dist band if necessary; currently use isofit's
    obs_envi = driver.Create(obs_outfile, xsize = cols, ysize = rows, bands = 10, 
                             eType = gdal.GDT_Float64, options = ['INTERLEAVE=BIL'])
    
    # File type must be float32 for rdn
    rdn_envi = driver.Create(rdn_outfile, xsize = cols, ysize = rows, bands = len(wvs), 
                             eType = gdal.GDT_Float32, options = ['INTERLEAVE=BIL'])
    
    # TODO: ensure don't need fwhm as rdn metadata
    rdn_envi.SetMetadataItem("wavelength", ("{"+", ".join(meta_wvs)+"}"), "ENVI")
    rdn_envi.SetMetadataItem("fwhm", ("{"+", ".join(fwhm)+"}"), "ENVI")

    # Convert solar zenith angle to radians
    u0 = np.cos(np.deg2rad(sza))
    rdn_dict = {}
    for i in range(len(rfls)):
        # Original units are reflectance, so convert to radiance in ISOFIT units [uW cm-2 nm-2 sr-1]
        # TODO: Figure out esd correction, bc these rdns are Lt*esd corr
        rdn_dict['Rdn_'+meta_wvs[i]] = rfls[i] * (((f0s[i])*u0)/np.pi)

    # Create ENVI files from dicts 
    add_bands(rdn_envi, rdn_dict, waves=wvs, width=fwhm, metadata=True)
    rdn_envi = None                                                             # Close to save file content

    loc_dict = {"Longitude":lon, "Latitude":lat,
                "Elevation (meters)":height}
    add_bands(loc_envi, loc_dict)
    loc_envi = None

    slope, aspect = calc_slope_aspect(height, lat, lon)
    cosi = calc_cosi(slope, aspect, sza, saa)    
    
    obs_dict = {"Pathlength (meters)":np.asarray(pathlength), 
                "Sensor Azimuth (0 - 360)":vaa,
                "Sensor Zenith (0 - 90)":vza,
                "Solar Azimuth (0 - 360)":saa,
                "Solar Zenith (0 - 90)":sza,
                "Phase angle (degrees)":phang, "Terrain Slope (degrees)":slope, 
                "Terrain Aspect":aspect, "Cosine(Solar Incidence Angle)":cosi, 
                "UTC Time (dec. hours)":utctime}
    add_bands(obs_envi, obs_dict)
    obs_envi = None

def add_bands(out_envi, rast_dict, waves=None, width=None, metadata=False) -> None:
    """
    Add bands to the ENVI object being created. No data value = np.NaN
    Requires:
        out_envi:       GDAL ENVI file object to create
        rast_dict:      Dictionary with keys=raster band names and val=value arrays
    Optional:
        waves:          Array of wavelengths for metadata. Default=None
        width:          Array of FWHM values for each wavelength to include in
                        metadata. Default=None
        metadata:       Boolean to write wavelength/fwhm metadata to ENVI file. 
                        Only required for rdn file, default=False
    Output:
        Writes raster bands to ENVI objects
    """
    # Start with first band
    band = 1
    # For each raster (key-val pair), create the band with necessary items
    for name in rast_dict.keys():
        rast = np.asarray(rast_dict[name])
        out_envi.GetRasterBand(band).WriteArray(rast)
        out_envi.GetRasterBand(band).SetDescription(name)
        out_envi.GetRasterBand(band).SetNoDataValue(np.NaN)
        if metadata:
            out_envi.GetRasterBand(band).SetMetadata({'wavelength':waves[band-1], 
                                                        'fwhm':width[band-1]})
        band += 1

def calc_cosi(slope, aspect, sza, saa) -> np.array:
    """
    Calculate the cosine of the incidence angle to include in obs ENVI file. 
    Cos(I) defined as the angle between the normal to the pixel surface and the 
    solar zenith direction
    Requires:
        slope:          array of slope values for each pixel (degrees)
        aspect:         array of aspect values for each pizel (degrees)
        sza:            array of solar zenith angle values (degrees)
        saa:            array of solar azimuth angle values (degrees)
    Output:
        cosi:           array of cos(incidence angle) values for each pixel (radians)
    """
    # TODO: Verify eqns
    sza_rad = np.deg2rad(sza)
    slope_rad = np.asarray(np.deg2rad(slope))
    raa = np.asarray(aspect) - saa
    raa_rad = np.deg2rad(raa)

    cosi = (np.cos(sza_rad)*np.cos(slope_rad)) + (np.sin(sza_rad)*np.sin(slope_rad)*np.cos(raa_rad))

    return np.asarray(cosi)

def calc_slope_aspect(height, lat, lon, outpath=os.getcwd()) -> tuple:
    """
    Calculates aspect in degress from an input DEM, in this case the loc
    ENVI file created in the nc2envi fcn. 
    Requires:
        height/lat/lon: geolocation arrays
    Optional:
        outpath:        Specific path to place output
    Returns:
        slope:          array of slope values for each pixel (degrees)
        aspect:         array of aspect values for each pizel (degrees)
                        no data vals are converted from -9999 to NaN
    """
    height = np.asarray(height)
    cols = height.shape[1]
    rows = height.shape[0]

    # Initialize DEM tif from input file
    dempath = join(outpath, "DEM.tif")
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(dempath, xsize=cols, ysize=rows, 
                       bands=1, eType=gdal.GDT_Float64)
    
    # Create ground control points for geolocation
    gcps = get_gcps(height, lat, lon, rows, cols)
    # Set spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)                                                    # TODO: EPSG okay?
    ds.SetGCPs(gcps, srs.ExportToWkt())
    ds.SetProjection(srs.ExportToWkt())
    # Height array will act as our DEM
    ds.GetRasterBand(1).WriteArray(height)
    ds = None

    # Create slope and aspect tifs from new DEM file
    outslope, outasp = join(outpath, "slope.tif"), join(outpath, "aspect.tif")
    slopefile = gdal.DEMProcessing(outslope, dempath, "slope", computeEdges=True)
    aspfile = gdal.DEMProcessing(outasp, dempath, "aspect", computeEdges=True)
    
    slopefile = None
    aspfile = None

    # Open and read in slope and aspect info
    with rasterio.open(outslope) as ds:
        slope=ds.read(1)
    slope[slope==-9999.000] = np.nan                                            # Set nodata value to nans

    with rasterio.open(outasp) as ds:
        aspect=ds.read(1)
    aspect[aspect==-9999.000] = np.nan

    # Delete unnecessary files
    os.remove(outslope)
    os.remove(outasp)
    os.remove(dempath)

    return slope, aspect

def calc_phase_angle(sza, saa, vza, vaa)->np.array:
    """
    Calculate phase angle. 
    should return phase angle in degrees 
    TODO: Check delphi calculation
    Requires:
        sza:        array of solar zenith angle (degrees)
        saa:        array of solar azimuth angle (degrees)
        vza:        array of view zenith angle (degrees)
        vaa:        array of view azimuth angle (degrees)
    Output:
        phang_deg:  array of phase angles (degrees)
    """
    # Convert degrees to radians
    sza_rad = np.deg2rad(sza)
    vza_rad = np.deg2rad(vza)

    # Calc rel. azimuth angle
    delphi = saa - vaa
    delphi = delphi % 360
    # Convert to radians
    raa_rad = np.deg2rad((delphi - 180))

    # Calculate phase angle
    phang_rad = np.arccos(np.cos(sza_rad)*np.cos(vza_rad) + np.sin(sza_rad)*np.sin(vza_rad)*np.cos(raa_rad))
    phang_deg = phang_rad * (180 / np.pi)
    return phang_deg

def calc_time(times: np.array, rows: int, cols: int)->np.array:
    """
    Convert time to utc hours since 00:00 on the day of observation for each pixel
    Currently each value in the time array is the time the line was observed.
    Each pixel is populated with an evenly spaced value based on the the start
    time of the next line and the number of pixels in the row
    Requires:
        times:          array of times for each line (row)
        rows/cols:      integer values for the dimensions of the raster
    Output:
        utctime:        array of time of observation for each pixel
    """
    # Read in times for each line 
    utcline = np.asarray([times[val] / 3600 for val in range(len(times))])
    ds = [abs(utcline[i] - utcline[i-1]) for i in range(len(utcline))]
    delt = np.mean(ds[1:])
    # Create utc time for each px instead of just each line 
    utctime = []
    l = 1
    while l < len(utcline):
        utcpx = np.linspace(utcline[l-1], utcline[l], num = cols)
        l += 1
        utctime.append(utcpx)
    # Calculate for the last line
    utctime.append(np.linspace(utcline[rows-1], (utcline[rows-1] - delt), num = cols))
    return np.asarray(utctime)

def get_gcps(height:np.array, lat:np.array, lon:np.array, rows:int, cols:int) -> list:
    """
    Set corner ground control points for minimal geolocation
    GCP Format: [map x-coordinate(longitude)], [map y-coordinate (latitude)], 
                [elevation], [image column index(x)], [image row index (y)]
    Requires:
        height/lat/lon: geolocation arrays
        rows/cols:      integer values for the dimensions of the raster
    Output:
        gcps:           list of four corner gcps
    """
    gcps = [gdal.GCP(float(lon[0][0]), float(lat[0][0]), float(height[0][0]), 0, 0),
            gdal.GCP(float(lon[0][-1]), float(lat[0][-1]), float(height[0][-1]), cols, 0),
            gdal.GCP(float(lon[-1][0]), float(lat[-1][0]), float(height[-1][0]), 0, rows),
            gdal.GCP(float(lon[-1][-1]), float(lat[-1][-1]), float(height[-1][-1]), cols, rows)]
    return gcps