#!/usr/bin/env python3

from pathlib import Path
import re
import json
import copy
import sys

import numpy as np
import pandas as pd
import spectral as sp
import matplotlib.pyplot as plt

assert len(sys.argv) > 1, "Please specify a JSON config file."

configfile = sys.argv[1]
with open(configfile, "r") as f:
    config = json.load(f)

outdir = Path(config["outdir"])

reflfiles = list(outdir.glob("**/estimated-reflectance"))
assert len(reflfiles) > 0, f"No reflectance files found in directory {outdir}"

true_refl_file = Path(config["reflectance_file"]).expanduser()
true_reflectance = sp.open_image(str(true_refl_file) + ".hdr")
true_waves = np.array(true_reflectance.metadata["wavelength"], dtype=float)
true_refl_m = true_reflectance.open_memmap()

windows = config["isofit"]["implementation"]["inversion"]["windows"]

def parse_dir(ddir):
    grps = {"directory": [str(ddir)]}
    for key in ["atm", "noise", "prior", "inversion"]:
        pat = f".*{key}_(.+?)" + r"(__|/|\Z)"
        match = re.match(pat, str(ddir))
        if match is not None:
            match = match.group(1)
        grps[key] = [match]
    for key in ["szen", "ozen", "saz", "oaz", "snr", "aod", "h2o"]:
        pat = f".*{key}_([0-9.]+)" + r"(__|/|\Z)"
        match = re.match(pat, str(ddir))
        if match is not None:
            match = float(match.group(1))
        grps[key] = [match]
    return pd.DataFrame(grps, index=[0])


info = pd.concat([parse_dir(x.parent) for x in reflfiles])\
         .reset_index(drop=True)


def mask_windows(data, waves, windows):
    inside_l = []
    for w in windows:
        inside_l.append(np.logical_and(waves >= w[0],
                                       waves <= w[1]))
    inside = np.logical_or.reduce(inside_l)
    d2 = copy.copy(data)
    d2[:, :, np.logical_not(inside)] = np.nan
    return d2


info["rmse"] = np.nan
info["bias"] = np.nan
info["rel_bias"] = np.nan
for i in range(info.shape[0]):
    ddir = Path(info["directory"][i])
    est_refl_file = ddir / "estimated-reflectance"
    est_refl = sp.open_image(str(est_refl_file) + ".hdr")
    est_refl_waves = np.array(est_refl.metadata["wavelength"], dtype=float)
    est_refl_m = est_refl.open_memmap()
    if est_refl_m.shape != true_refl_m.shape:
        true_resample = np.zeros_like(est_refl_m)
        for r in range(true_resample.shape[0]):
            for c in range(true_resample.shape[1]):
                true_resample[r, c, :] = np.interp(
                    est_refl_waves,
                    true_waves,
                    true_refl_m[r, c, :]
                )
    else:
        true_resample = true_refl_m
    est_refl_m2 = mask_windows(est_refl_m, est_refl_waves, windows)
    bias = est_refl_m2 - true_resample
    rmse = np.sqrt(np.nanmean(bias**2))
    mean_bias = np.nanmean(bias)
    rel_bias = bias / true_resample
    mean_rel_bias = np.nanmean(rel_bias)
    info.loc[i, "rmse"] = rmse
    info.loc[i, "bias"] = mean_bias
    info.loc[i, "rel_bias"] = mean_rel_bias
    # Bias by wavelength
    bias_wl = np.nanmean(bias, axis=(0, 1))
    bias_wl_q = np.nanquantile(bias, (0.05, 0.95), axis=(0, 1))
    plt.axhline(y=0, color="gray")
    plt.plot(est_refl_waves, bias_wl, "k-")
    plt.plot(est_refl_waves, np.transpose(bias_wl_q), "k--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Bias (Estimate - True; 90% CI)")
    plt.savefig(ddir / "bias.png")

print("Simulations sorted by RMSE (lowest first)")
print(info.sort_values("rmse"))

info.to_csv(outdir / "summary.csv")
