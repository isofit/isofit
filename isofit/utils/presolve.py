import os
from glob import glob
import ray
import numpy as np
from spectral.io import envi

from isofit.core.common import envi_header
from isofit.configs import configs
from isofit.inversion.inverse_simple import invert_simple
from isofit.core.forward import ForwardModel
from isofit.core.fileio import IO
from isofit.core.geometry import Geometry


def presolve_atm(paths, working_directory, n_chunks=50):
    """Coarse solve of H2O and AOT of the segmented image."""

    config = configs.create_new_config(
        glob(os.path.join(working_directory, "config", "") + "*_isofit.json")[0]
    )
    fm = ForwardModel(config)
    esd = IO.load_esd()

    rdn_hdr = envi_header(paths.rdn_subs_path)
    rdn = envi.open(rdn_hdr, paths.rdn_subs_path).open_memmap()

    chunk_size = max(1, rdn.shape[0] // n_chunks)
    chunk_indices = [
        range(start, min(start + chunk_size, rdn.shape[0]))
        for start in range(0, rdn.shape[0], chunk_size)
    ]

    futures = [
        worker.remote(
            idx_chunk,
            paths.rdn_subs_path,
            paths.loc_subs_path,
            paths.obs_subs_path,
            ray.put(fm),
            ray.put(esd),
        )
        for idx_chunk in chunk_indices
    ]

    wv_array = np.empty((rdn.shape[0], 1, 1), dtype=np.float32)
    aot_array = np.empty((rdn.shape[0], 1, 1), dtype=np.float32)

    for idx_chunk, result_chunk in ray.get(futures):
        wv_array[idx_chunk, 0, 0] = result_chunk[0]
        aot_array[idx_chunk, 0, 0] = result_chunk[1]

    return wv_array, aot_array


@ray.remote
def worker(idx_chunk, rdn_path, loc_path, obs_path, fm, esd):
    rdn = envi.open(envi_header(rdn_path), rdn_path).open_memmap()
    loc = envi.open(envi_header(loc_path), loc_path).open_memmap()
    obs = envi.open(envi_header(obs_path), obs_path).open_memmap()

    for h2oname in ["H2OSTR", "h2o"]:
        if h2oname in fm.RT.statevec_names:
            wv_idx = fm.RT.statevec_names.index(h2oname)
            break
    else:
        raise ValueError("Water vapor not found in RT names.")

    aot_idx = fm.RT.statevec_names.index("AOT550")

    wv_chunk = np.empty(len(idx_chunk), dtype=np.float32)
    aot_chunk = np.empty(len(idx_chunk), dtype=np.float32)

    for i, idx in enumerate(idx_chunk):
        geom = Geometry(obs=obs[idx, 0, :], loc=loc[idx, 0, :], esd=esd, svf=1.0)
        x_center = invert_simple(fm, rdn[idx, 0, :], geom)
        wv_chunk[i] = x_center[fm.idx_RT][wv_idx]
        aot_chunk[i] = x_center[fm.idx_RT][aot_idx]

    return idx_chunk, (wv_chunk, aot_chunk)
