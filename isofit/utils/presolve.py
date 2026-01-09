import ray
import numpy as np
from spectral.io import envi

from isofit.core.common import envi_header
from isofit.configs import configs
from isofit.inversion.inverse_simple import invert_simple
from isofit.core.forward import ForwardModel
from isofit.core.fileio import IO
from isofit.core.geometry import Geometry


def presolve_atm(
    input_radiance, input_loc, input_obs, paths, use_superpixels=True, n_chunks=50
):
    """Heuristic solve of the segmented (or original image) for aerosol optical depth and water vapor."""
    # Load forward model (creates presolve LUT here if not yet created)
    config = configs.create_new_config(paths.isofit_full_config_path)
    fm = ForwardModel(config)
    esd = IO.load_esd()

    input_radiance = paths.rdn_subs_path if use_superpixels else input_radiance
    input_loc = paths.loc_subs_path if use_superpixels else input_loc
    input_obs = paths.obs_subs_path if use_superpixels else input_obs

    # Determine chunking for ray
    loc = envi.open(envi_header(input_loc), input_loc).open_memmap()
    if use_superpixels:
        n = loc.shape[0]
    else:
        n = loc.shape[0] * loc.shape[1]

    chunk_size = max(1, n // n_chunks)
    chunk_indices = [
        range(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)
    ]

    # Check water vapor and aot index
    wv_idx = next(
        (i for i, n in enumerate(fm.RT.statevec_names) if n.lower().startswith("h2o")),
        None,
    )
    aot_idx = next(
        (i for i, n in enumerate(fm.RT.statevec_names) if n.lower().startswith("aot")),
        None,
    )
    if wv_idx is None or aot_idx is None:
        raise ValueError("Required RT state vector not found.")

    futures = [
        worker.remote(
            idx_chunk,
            input_radiance,
            input_loc,
            input_obs,
            ray.put(fm),
            ray.put(esd),
            use_superpixels,
            wv_idx,
            aot_idx,
        )
        for idx_chunk in chunk_indices
    ]

    if use_superpixels:
        wv_array = np.empty((loc.shape[0], 1, 1), dtype=np.float32)
        aot_array = np.empty((loc.shape[0], 1, 1), dtype=np.float32)
    else:
        wv_array = np.empty((loc.shape[0], loc.shape[1], 1), dtype=np.float32)
        aot_array = np.empty((loc.shape[0], loc.shape[1], 1), dtype=np.float32)

    for idx_chunk, result_chunk in ray.get(futures):
        if use_superpixels:
            wv_array[idx_chunk, 0, 0] = result_chunk[0]
            aot_array[idx_chunk, 0, 0] = result_chunk[1]
        else:
            n_cols = loc.shape[1]
            for i, idx in enumerate(idx_chunk):
                row = idx // n_cols
                col = idx % n_cols
                wv_array[row, col, 0] = result_chunk[0][i]
                aot_array[row, col, 0] = result_chunk[1][i]

    return wv_array, aot_array


@ray.remote
def worker(
    idx_chunk, rdn_path, loc_path, obs_path, fm, esd, use_superpixels, wv_idx, aot_idx
):
    """Calls invert_simple and stores the aot and wv values."""
    rdn = envi.open(envi_header(rdn_path), rdn_path).open_memmap()
    loc = envi.open(envi_header(loc_path), loc_path).open_memmap()
    obs = envi.open(envi_header(obs_path), obs_path).open_memmap()

    wv_chunk = np.empty(len(idx_chunk), dtype=np.float32)
    aot_chunk = np.empty(len(idx_chunk), dtype=np.float32)

    for i, idx in enumerate(idx_chunk):
        if use_superpixels:
            row, col = idx, 0
        else:
            row = idx // rdn.shape[1]
            col = idx % rdn.shape[1]

        geom = Geometry(obs=obs[row, col, :], loc=loc[row, col, :], esd=esd, svf=1.0)
        x_center = invert_simple(fm, rdn[row, col, :], geom)
        wv_chunk[i] = x_center[fm.idx_RT][wv_idx]
        aot_chunk[i] = x_center[fm.idx_RT][aot_idx]

    return idx_chunk, (wv_chunk, aot_chunk)
