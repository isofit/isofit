from __future__ import annotations

import logging
import multiprocessing
import re
import time
from datetime import datetime

import click
import numpy as np
from scipy import ndimage
from scipy.io import loadmat
from scipy.linalg import norm
from spectral import envi

from isofit import ray
from isofit.core.common import envi_header, resample_spectrum, svd_inv
from isofit.core.fileio import IO, initialize_output, write_bil_chunk
from isofit.core.geometry import Geometry
from isofit.data import env


class Component:
    def __init__(self, surface_files, mapping):

        # Combine all potential surface files
        self.model_dict = load_surface_mat(surface_files)
        self.components = list(zip(self.model_dict["means"], self.model_dict["covs"]))
        self.n_comp = len(self.components)
        self.wl = self.model_dict["wl"][0]
        self.n_wl = len(self.wl)

        self.surface_types = self.model_dict.get("surface_types", [])
        self.mapping = mapping

        normalize = self.model_dict["normalize"]
        if normalize == "Euclidean":
            self.norm = lambda r: norm(r)
        elif normalize == "RMS":
            self.norm = lambda r: np.sqrt(np.mean(pow(r, 2)))
        elif self.normalize == "None":
            self.norm = lambda r: 1.0
        else:
            raise ValueError("Unrecognized Normalization: %s\n" % normalize)

        refwl = np.squeeze(self.model_dict["refwl"])
        refwl = refwl[(refwl < 900) | (refwl > 2000)]
        idx_ref = [np.argmin(abs(self.wl - w)) for w in np.squeeze(refwl)]
        self.idx_ref = np.array(idx_ref)

        self.Covs, self.Cinvs, self.mus = [], [], []
        for i in range(self.n_comp):
            Cov = self.components[i][1]
            self.Covs.append(np.array([Cov[j, self.idx_ref] for j in self.idx_ref]))
            self.Cinvs.append(svd_inv(self.Covs[-1]))
            self.mus.append(self.components[i][0][self.idx_ref])

    def pickClosest(self, x, geom):
        lamb_ref = x[self.idx_ref]

        lamb_ref = (lamb_ref - np.min(lamb_ref)) / (np.max(lamb_ref) - np.min(lamb_ref))

        mds = []
        for ci in range(self.n_comp):
            ref_mu = self.mus[ci]
            ref_mu = (ref_mu - np.min(ref_mu)) / (np.max(ref_mu) - np.min(ref_mu))
            mds.append(sum(pow(lamb_ref - ref_mu, 2)))
        closest = np.argmin(mds)

        surface_type = self.surface_types[closest].strip()
        surface_idx = [i for i, val in enumerate(self.mapping) if val == surface_type]
        surface_idx = surface_idx[0] if len(surface_idx) else -9999

        return surface_idx


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(
        self,
        rdn_file: str,
        obs_file: str,
        loc_file: str,
        out_file: str,
        surface_path: str,
        mapping: list,
        wl: np.ndarray,
        fwhm: np.ndarray,
        dayofyear: int,
        irr_file: str,
        loglevel: str,
        logfile: str,
    ):
        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=loglevel,
            filename=logfile,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )

        self.rdn = envi.open(envi_header(rdn_file)).open_memmap(interleave="bip")
        self.loc = envi.open(envi_header(loc_file)).open_memmap(interleave="bip")
        self.obs = envi.open(envi_header(obs_file)).open_memmap(interleave="bip")
        self.out_file = out_file
        self.out = envi.open(envi_header(out_file)).open_memmap(
            interleave="bip", writable=True
        )

        self.component = Component(surface_path, mapping)
        self.esd = IO.load_esd()
        self.dayofyear = dayofyear

        self.wl = wl
        self.fwhm = fwhm
        self.solar_irr = self.solarIrradiance(irr_file)

    def solarIrradiance(self, irr_file):
        irr = np.loadtxt(irr_file, comments="#")
        iwl, irr = irr.T
        if iwl[0] > 100:
            iwl = iwl / 1000.0
        irr = irr / 10.0  # convert, uW/nm/cm2
        irr_factor = self.esd[self.dayofyear - 1, 1]
        irr = irr / irr_factor**2  # consider solar distance

        return resample_spectrum(irr, iwl, self.wl, self.fwhm)

    def run_lines(self, startstop):
        start_line, stop_line = startstop
        output = self.out[start_line:stop_line, ...]
        for r in range(start_line, stop_line):
            for c in range(output.shape[1]):
                meas = self.rdn[r, c, :]
                geom = Geometry(
                    obs=self.obs[r, c, :], loc=self.loc[r, c, :], esd=self.esd
                )
                coszen = np.cos(np.deg2rad(geom.solar_zenith))

                num = meas * np.pi
                denom = self.solar_irr * coszen

                x = num / denom
                output[r - start_line, c, :] = self.component.pickClosest(x, geom)

            unique, counts = np.unique(output[r - start_line, ...], return_counts=True)
            logging.debug(f"Elements: {unique}")
            logging.debug(f"Counts: {counts}")

            write_bil_chunk(
                np.swapaxes(output, 1, 2),
                self.out_file,
                start_line,
                (self.out.shape[0], self.out.shape[1], 1),
            )


def load_surface_mat(
    surface_files,
    keys_to_combine=[
        "means",
        "covs",
        "attribute_means",
        "attribute_covs",
        "surface_types",
    ],
):
    # CLI will pass .json as string. Convert to temp dict
    if isinstance(surface_files, str):
        surface_files = {"cli_input": surface_files}

    for i, (name, surface_file) in enumerate(surface_files.items()):
        surface_model_dict = loadmat(surface_file)
        if not i:
            model_dict = surface_model_dict
        else:
            for key in keys_to_combine:
                model_dict[key] = np.concatenate(
                    [model_dict[key], surface_model_dict[key]], axis=0
                )

    return model_dict


def filter_image(out, thresh=100):
    """
    Temporary function to clean the image.
    Memory intensive.

    Could try to make this recursive by nesting the bottom loop into the top loop
    """
    # Don't do any filtering if array is uniform
    if len(np.unique(out)) == 1:
        return out

    masks = []
    for i in np.unique(out):
        temp = out.copy()
        temp[temp == i] = 9999
        temp[temp < 9999] = 0
        temp[temp == 9999] = 1
        label, n = ndimage.label(temp)
        sizes = ndimage.sum(temp, label, range(n + 1))
        mask = sizes >= thresh
        masks.append(mask[label])

    for i, mask in enumerate(masks):
        if not i:
            final = mask.astype(int) * (i + 1)
        else:
            final += mask.astype(int) * (i + 1)

    label, n = ndimage.label(final == 0)
    for i in np.unique(label):
        if not i:
            continue

        temp = label.copy()
        temp[temp != i] = 0
        temp[temp > 0] = 1

        vals, counts = np.unique(
            final[np.where(ndimage.binary_dilation(temp).astype(int) - temp)],
            return_counts=True,
        )
        final[label == i] = vals[np.argmax(counts)]

    return final - 1


def multicomponent_classification(
    rdn_file: str,
    obs_file: str,
    loc_file: str,
    out_file: str,
    surface_files: str,
    n_cores: int = -1,
    dayofyear: int = None,
    wl_file: str = None,
    irr_file: str = None,
    mapping: list = None,
    clean: bool = False,
    thresh: int = 100,
    ray_address: str = None,
    ray_redis_password: str = None,
    ray_temp_dir=None,
    ray_ip_head=None,
    loglevel: str = "INFO",
    logfile: str = None,
):
    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=loglevel,
        filename=logfile,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    # Get day of year from rdn string
    if not dayofyear:
        match = re.search("([0-9]{8}t[0-9]{6})", rdn_file)
        if match:
            dt = datetime.strptime(match.group(), "%Y%m%dt%H%M%S")
            dayofyear = dt.timetuple().tm_yday
        else:
            logging.error("Could not find day of year from path")
            raise ValueError("Could not find day of year from path")

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()

    # Construct the output File
    rdn_ds = envi.open(envi_header(rdn_file))
    rdns = rdn_ds.shape
    output_metadata = rdn_ds.metadata

    # Get wavelength
    if wl_file:
        # Assumes a structure where
        # column 0: idx
        # column 1: wl
        # column 2: fwhm
        wl = np.loadtxt(wl_file)
        fwhm = wl[:, 2]
        wl = wl[:, 1]
    else:
        wl = np.array(rdn_ds.metadata.get("wavelength", [])).astype(float)
        fwhm = np.array(rdn_ds.metadata.get("fwhm", [])).astype(float)

        if not len(wl) or not len(fwhm):
            message = (
                "No wavelength file given and rdn file does "
                "not contain wavelength or fwhm information"
            )
            logging.error(message)
            raise KeyError(message)

    # Delete rdn_ds now that we don't need it
    del rdn_ds

    # Check units of wavelength
    if wl[0] > 100:
        logging.info("Wavelength units of nm inferred...converting to microns")
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    # Check to see if irradiance file was passed
    irr_path = [
        "examples",
        "20151026_SantaMonica",
        "data",
        "prism_optimized_irr.dat",
    ]
    irr_file = irr_file if irr_file else str(env.path(*irr_path))

    # The "mapping" is how the program moves between a int-classification
    # And the surface model
    if not mapping:
        model_dict = load_surface_mat(surface_files)
        surface_types = model_dict.get("surface_types", [])
        del model_dict

        if len(surface_types):
            mapping = []
            for i in surface_types:
                if isinstance(i, np.ndarray) or isinstance(i, list):
                    i = i[0].strip()
                else:
                    i = i.strip()
                if i not in mapping:
                    mapping.append(i)
        else:
            logging.error("No surface mapping provided")
            raise ValueError("No surface mapping provided")

    output = initialize_output(
        output_metadata,
        out_file,
        (rdns[0], 1, rdns[1]),
        ["emit pge input files", "wavelength", "fwhm"],
        interleave="bil",
        bands="1",
        band_names=["Classification"],
        description=("Per-pixel multicomponent classification"),
        mapping=mapping,
    )

    # Ray initialization
    ray_dict = {
        "ignore_reinit_error": True,
        "local_mode": n_cores == 1,
        "address": ray_address,
        "include_dashboard": False,
        "_temp_dir": ray_temp_dir,
        "_redis_password": ray_redis_password,
    }
    ray.init(**ray_dict)
    if n_cores == 1:
        n_workers = n_cores + 1
    else:
        n_workers = n_cores

    line_breaks = np.linspace(0, rdns[0], n_workers, dtype=int)
    line_breaks = [
        (line_breaks[n], line_breaks[n + 1]) for n in range(len(line_breaks) - 1)
    ]

    wargs = [
        ray.put(obs)
        for obs in (
            rdn_file,
            obs_file,
            loc_file,
            out_file,
            surface_files,
            mapping,
            wl,
            fwhm,
            dayofyear,
            irr_file,
            loglevel,
            logfile,
        )
    ]
    workers = ray.util.ActorPool([Worker.remote(*wargs) for _ in range(n_workers)])

    start_time = time.time()
    res = list(workers.map_unordered(lambda a, b: a.run_lines.remote(b), line_breaks))
    total_time = time.time() - start_time

    logging.info(
        f"Multicomponent classification complete.  {round(total_time,2)}s total, "
        f"{round(rdns[0]*rdns[1]/total_time,4)} spectra/s, "
        f"{round(rdns[0]*rdns[1]/total_time/n_workers,4)} spectra/s/core"
    )

    if clean:
        logging.info("Filtering classification image." f"Using thresh: {thresh}")
        out = envi.open(envi_header(out_file)).open_memmap(
            interleave="bip", writable=True
        )
        out_filter = filter_image(out.copy(), thresh)
        out[...] = out_filter


@click.command(
    name="multicomponent_classification",
    help=multicomponent_classification.__doc__,
    no_args_is_help=True,
)
@click.argument("rdn_file")
@click.argument("obs_file")
@click.argument("loc_file")
@click.argument("out_file")
@click.argument("surface_files")
@click.option("--n_cores", default=-1)
@click.option("--wl_file", default=None)
@click.option("--irr_file", default=None)
@click.option("--mapping", default=None)
@click.option("--clean", is_flag=True, default=False)
@click.option("--thresh", default=100)
@click.option("--ray_address", default=None)
@click.option("--ray_redis_password", default=None)
@click.option("--ray_temp_dir", default=None)
@click.option("--ray_ip_head", default=None)
@click.option("--loglevel", default="INFO")
@click.option("--logfile", default=None)
def cli(**kwargs):
    multicomponent_classification(**kwargs)
    click.echo("Done")
