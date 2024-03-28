"""
Apply a (possibly multi-file) per-pixel spatial reference.

Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
"""

import argparse
import numpy as np
import pandas as pd
from spectral.io import envi
import logging
from isofit import ray
from typing import List
import os
import multiprocessing
from isofit.core.common import envi_header

GLT_NODATA_VALUE=-9999
CRITERIA_NODATA_VALUE = -9999


def main():
    parser = argparse.ArgumentParser(description='Integrate multiple GLTs with a mosaicing rule')
    parser.add_argument('rawspace_file',
                        help='filename of rawspace source file or, in the case of a mosaic_glt, a text-file list of raw space files')
    parser.add_argument('glt_file')
    parser.add_argument('output_filename')
    parser.add_argument('--band_numbers', nargs='+', type=int, default=-1,
                        help='list of 0-based band numbers, or -1 for all')
    parser.add_argument('--n_cores', type=int, default=-1)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--run_with_missing_files', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ip_head', type=str)
    parser.add_argument('--redis_password', type=str)
    parser.add_argument('--one_based_glt', type=int, choices=[0, 1], default=0)
    parser.add_argument('--mosaic', type=int, choices=[0, 1], default=0)
    global GLT_NODATA_VALUE
    parser.add_argument('--glt_nodata_value', type=float, default=GLT_NODATA_VALUE)
    args = parser.parse_args()

    # Set up logging per arguments
    if args.log_file is None:
        logging.basicConfig(format='%(message)s', level=args.log_level)
    else:
        logging.basicConfig(format='%(message)s', level=args.log_level, filename=args.log_file)

    args.one_based_glt = args.one_based_glt == 1
    args.run_with_missing_files = args.run_with_missing_files == 1
    args.mosaic = args.mosaic == 1
    GLT_NODATA_VALUE = args.glt_nodata_value

    # Log the current time
    logging.info('Starting apply_glt, arguments given as: {}'.format(args))

    # Open the GLT dataset
    glt_dataset = envi.open(envi_header(args.glt_file))
    glt = glt_dataset.open_memmap(writeable=False, interleave='bip')

    if args.mosaic:
        rawspace_files = np.squeeze(np.array(pd.read_csv(args.rawspace_file, header=None)))
    else:
        rawspace_files = [args.rawspace_file]

    for _ind in range(len(rawspace_files)):
        first_file = envi_header(rawspace_files[_ind])
        if os.path.isfile(first_file):
            break
    first_file_dataset = envi.open(first_file)

    if args.band_numbers == -1:
        output_bands = np.arange(int(first_file_dataset.metadata['bands']))
    else:
        output_bands = np.array(args.band_numbers)

    outmeta = first_file_dataset.metadata.copy()
    outmeta["lines"] = glt_dataset.metadata["lines"]
    outmeta["samples"] = glt_dataset.metadata["samples"]
    outmeta["map info"] = glt_dataset.metadata["map info"]
    outmeta["coordinate system string"] = glt_dataset.metadata["coordinate system string"]
    if "band names" in first_file_dataset.metadata.keys():
        outmeta["band names"] = [first_file_dataset.metadata["band names"][i] for i in output_bands]
    if "wavelength" in first_file_dataset.metadata.keys():
        outmeta["wavelength"] = [first_file_dataset.metadata["wavelength"][i] for i in output_bands]
    if "fwhm" in first_file_dataset.metadata.keys():
        outmeta["fwhm"] = [first_file_dataset.metadata["fwhm"][i] for i in output_bands]

    outDataset = envi.create_image(args.output_filename, outmeta, ext='', force=True)
    del outDataset

    if args.n_cores == -1:
        args.n_cores = multiprocessing.cpu_count()

    rayargs = {'address': args.ip_head,
               '_redis_password': args.redis_password,
               'local_mode': args.n_cores == 1}
    if args.n_cores < 40:
        rayargs['num_cpus'] = args.n_cores
    ray.init(**rayargs)

    jobs = []
    for idx_y in range(glt.shape[0]):
        jobs.append(apply_mosaic_glt_line.remote(args.glt_file,
                                                 args.output_filename,
                                                 rawspace_files,
                                                 output_bands,
                                                 idx_y,
                                                 args))
    rreturn = [ray.get(jid) for jid in jobs]
    ray.shutdown()

    # Log final time and exit
    logging.info('GLT application complete, output available at: {}'.format(args.output_filename))


def _write_bil_chunk(dat: np.array, outfile: str, line: int, shape: tuple, dtype: str = 'float32') -> None:
    """
    Write a chunk of data to a binary, BIL formatted data cube.
    Args:
        dat: data to write
        outfile: output file to write to
        line: line of the output file to write to
        shape: shape of the output file
        dtype: output data type

    Returns:
        None
    """
    outfile = open(outfile, 'rb+')
    outfile.seek(line * shape[1] * shape[2] * np.dtype(dtype).itemsize)
    outfile.write(dat.astype(dtype).tobytes())
    outfile.close()


@ray.remote(num_cpus=1)
def apply_mosaic_glt_line(glt_filename: str, output_filename: str, rawspace_files: List, output_bands: np.array,
                          line_index: int, args: List):
    """
    Create one line of an output mosaic in mapspace
    Args:
        glt_filename: pre-built single or mosaic glt
        output_filename: output destination, assumed to location where a pre-initialized raster exists
        rawspace_files: list of rawspace input locations
        output_bands: array-like of bands to use from the rawspace file in the output
        line_index: line of the glt to process
    Returns:
        None
    """

    logging.basicConfig(format='%(message)s', level=args.log_level, filename=args.log_file)

    glt_dataset = envi.open(glt_filename + '.hdr')
    glt = glt_dataset.open_memmap(writeable=False, interleave='bip')

    if line_index % 100 == 0:
        logging.info('Beginning application of line {}/{}'.format(line_index, glt.shape[0]))

    glt_line = np.squeeze(glt[line_index, ...]).copy()
    valid_glt = np.all(glt_line != GLT_NODATA_VALUE, axis=-1)

    glt_line[valid_glt, 1] = np.abs(glt_line[valid_glt, 1])
    glt_line[valid_glt, 0] = np.abs(glt_line[valid_glt, 0])
    glt_line[valid_glt, -1] = glt_line[valid_glt, -1]

    if args.one_based_glt:
        glt_line[valid_glt, :] = glt_line[valid_glt, :] - 1

    if np.sum(valid_glt) == 0:
        return

    if args.mosaic:
        un_file_idx = np.unique(glt_line[valid_glt, -1])
    else:
        un_file_idx = [0]

    output_dat = np.zeros((glt.shape[1], len(output_bands)), dtype=np.float32) - 9999
    for _idx in un_file_idx:
        if os.path.isfile(rawspace_files[_idx]):
            rawspace_dataset = envi.open(rawspace_files[_idx] + '.hdr')
            rawspace_dat = rawspace_dataset.open_memmap(interleave='bip')

            if args.mosaic:
                linematch = np.logical_and(glt_line[:, -1] == _idx, valid_glt)
            else:
                linematch = valid_glt

            if np.sum(linematch) > 0:
                output_dat[linematch, :] = rawspace_dat[
                    glt_line[linematch, 1][:, None], glt_line[linematch, 0][:, None], output_bands[None, :]].copy()

    _write_bil_chunk(np.transpose(output_dat), output_filename, line_index,
                     (glt.shape[0], len(output_bands), glt.shape[1]))


if __name__ == "__main__":
    main()
