"""
Example Usage:
python run.py
"""
import argparse
import cProfile
import logging
import os
import pstats
import requests
import subprocess
import sys
import zipfile

from glob      import glob
from importlib import reload
from io        import BytesIO

import isofit

from isofit.utils.multisurface_oe import main as multi_oe


logging.basicConfig(format='%(levelname)s | %(message)s', level=logging.DEBUG)

Logger    = logging.getLogger('isofit/examples/profiling_cube')
ISOFIT    = '/'.join(isofit.__file__.split('/')[:-2])
EX_DIR    = 'examples/profiling_cube'
FILE_BASE = 'ang20170323t202244'
URLS      = {
    'small' : 'https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/small_chunk.zip',
    'medium': 'https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/medium_chunk.zip'
}


def download(size):
    """
    Checks if the rdn, loc, and obs files for this size_chunk are present. If
    not, attempt to download and extract the zip they source from.

    Parameters
    ----------
    size: str
        Which chunk size to download for, such as 'small', 'medium'

    Returns
    -------
    files: list
        List of files [rdn, loc, obs]
    """
    files = glob(f'{size}_chunk/{FILE_BASE}_rdn_*[!.hdr]') \
          + glob(f'{size}_chunk/{FILE_BASE}_loc_*[!.hdr]') \
          + glob(f'{size}_chunk/{FILE_BASE}_obs_*[!.hdr]')

    if len(files) != 3:
        Logger.info('Downloading data')

        req = requests.get(URLS[size])
        if req:
            with zipfile.ZipFile(BytesIO(req.content)) as zip:
                zip.extractall()
        else:
            Logger.error(f'Failed to download {size}_chunk data with HTTP error code: {req.status_code}')

        # Try again
        files = glob(f'{size}_chunk/{FILE_BASE}_rdn_*[!.hdr]') \
              + glob(f'{size}_chunk/{FILE_BASE}_loc_*[!.hdr]') \
              + glob(f'{size}_chunk/{FILE_BASE}_obs_*[!.hdr]')

    if len(files) != 3:
        Logger.error('Not all input files are found')
        return

    return files

def profile(args, output=None):
    """
    Profiles calling the multisurface_oe:main().

    Parameters
    ----------
    args: list
        Arguments to be passed to multisurface_oe:main()
    output: str, default=None
        Path to an output file
    """
    profiler = cProfile.Profile()
    profiler.enable()

    # Make sure the module is a fresh instance between runs
    reload(isofit.utils.multisurface_oe)
    multi_oe(args)

    profiler.disable()

    if output:
        stats = pstats.Stats(profiler)
        stats.dump_stats(output)

def run(config, workdir, files, output=None, n=1):
    """
    Performs N many runs of multisurface_oe.

    Parameters
    ----------
    config: str
        Parameter for multisurface_oe
    workdir: str
        Parameter for multisurface_oe
    files: list
        Filepaths list for the rdn, loc, and obs files, in that order
    output: str, default=None
        Directory to output profiling.dat files
    n: int, default=1
        Number of profiling runs to repeat
    """
    file = None
    for i in range(1, n+1):
        Logger.info(f'Starting run {i}/{n}')

        if output:
            file = f'{output}/run_{i}.dat'

        # Make sure these directories are clean
        for file in glob(f'{workdir}/lut_h2o/*'):
            os.remove(file)
        for file in glob(f'{workdir}/output/*'):
            os.remove(file)

        profile(
            args   = files + [workdir, config, '--logging_level', 'DEBUG'],
            output = file
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p', '--path',     type    = str,
                                            metavar = '/path/to/isofit',
                                            help    = 'Path to the ISOFIT directory'
    )
    parser.add_argument('-o', '--output',   type    = str,
                                            metavar = '/path/to/profiling/outputs/',
                                            default = './profile_results',
                                            help    = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--size',     choices = ['small', 'medium'],
                                            default = 'small',
                                            help    = 'Size of chunk to run'
    )
    parser.add_argument('-m', '--method',   choices = ['mlg', 'rg', 'nds'],
                                            default = 'mlg',
                                            help    = 'Interpolation style to choose from'
    )
    parser.add_argument('-n', '--run_n',    type    = int,
                                            default = 1,
                                            help    = 'Number of runs to do'
    )
    parser.add_argument('-dc', '--disable_cleanup',
                                            action  = 'store_true',
                                            help    = 'Disables cleaning up the files after finishing'
    )
    args = parser.parse_args()

    if not os.path.isdir(f'{args.size}_chunk'):
        Logger.error(f'Missing directory {args.size}_chunk')
        sys.exit(1)
    dir = f'{args.size}_chunk'

    config = f'{dir}/configs/{args.method}.yml'
    if not os.path.isfile(config):
        Logger.error(f'Missing config file {config}')
        sys.exit(2)


    # Attempt to create the output directory if it does not exist
    output = args.output
    if not os.path.exists(output):
        Logger.warning('Output directory does not exist, attempting to fallback to this directory')
        output = './profile_results'
        if not os.path.exists(output):
            os.mkdir(output)
            Logger.info(f'Output directory set to: {output}')

    # Make sure we've downloaded the data already
    files = download(args.size)

    if not files:
        sys.exit(3)

    os.environ['ISOFIT_DEBUG'] = '1'

    # Select the correct config file and run it
    run(
        config  = config,
        files   = files,
        output  = output,
        workdir = dir,
        n       = args.run_n
    )

    if not args.disable_cleanup:
        file = f'{dir}.zip'
        if os.path.exists(file):
            Logger.info(f'Removing zip file: {file}')
            os.remove(file)
