"""
Nimrod Carmon's research code as-is for comparison
"""
import re

import numpy as np


def how_many_bands(file_path):

    num_bands = 0
    previous_band = 0
    number_pattern = re.compile(r"^[1-9]\d{2,}(\.\d+)?.*$")

    with open(file_path, "r") as file:
        for line in file:

            line = line.strip()  # Remove leading/trailing whitespaces
            if not line or not number_pattern.match(line):
                continue  # Skip empty lines or lines not matching the pattern

            current_band = float(
                line.split()[0]
            )  # Assumes numbers are in the first column

            if current_band < previous_band:
                break
            if current_band > 300 and current_band != previous_band:
                num_bands += 1
                previous_band = current_band

    return num_bands


"""
SOLZEN is in filename, needed for calculating coszen
"""

# need to complete the dict in so it replaces the self object here
def load_chn_single(infile, multipart):
    """Load a '.chn' output file and parse critical coefficient vectors.

    These are:
        * wl      - wavelength vector
        * sol_irr - solar irradiance
        * sphalb  - spherical sky albedo at surface
        * transm  - diffuse and direct irradiance along the
                    sun-ground-sensor path
        * transup - transmission along the ground-sensor path only


    If the "multipart transmittance" option is active, we will use
    a combination of three MODTRAN runs to estimate the following
    additional quantities:
        * t_down_dir - direct downwelling transmittance
        * t_down_dif - diffuse downwelling transmittance
        * t_up_dir   - direct upwelling transmittance
        * t_up_dif   - diffuse upwelling transmittance

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Be careful with these! They are to be used only by the
    modtran_tir functions because MODTRAN must be run with a
    reflectivity of 1 for them to be used in the RTM defined
    in radiative_transfer.py.

    * thermal_upwelling - atmospheric path radiance
    * thermal_downwelling - sky-integrated thermal path radiance
        reflected off the ground and back into the sensor.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    We parse them one wavelength at a time."""

    multipart_transmittance = multipart
    test_rfls = [0, 0.1, 0.5]
    coszen = np.cos(
        np.deg2rad(
            float(infile.strip(".chn").split("/")[-1].split("_")[0].split("-")[1])
        )
    )

    nwl = how_many_bands(infile)

    with open(infile) as f:
        sols, transms, sphalbs, wls, rhoatms, transups = [], [], [], [], [], []
        t_down_dirs, t_down_difs, t_up_dirs, t_up_difs = [], [], [], []
        grnd_rflts_1, drct_rflts_1, grnd_rflts_2, drct_rflts_2 = [], [], [], []
        transm_dirs, transm_difs, widths = [], [], []
        lp_0, lp_1, lp_2 = [], [], []
        thermal_upwellings, thermal_downwellings = [], []
        lines = f.readlines()
        nheader = 5
        # pdb.set_trace()
        # Mark header and data segments
        case = -np.ones(nheader * 3 + nwl * 3)
        case[nheader : (nheader + nwl)] = 0
        case[(nheader * 2 + nwl) : (nheader * 2 + nwl * 2)] = 1
        case[(nheader * 3 + nwl * 2) : (nheader * 3 + nwl * 3)] = 2

        for i, line in enumerate(lines):

            # exclude headers
            if case[i] < 0:
                continue

            try:
                # Columns 1 and 2 can touch for large datasets.
                # Since we don't care about the values, we overwrite the
                # character to the left of column 1 with a space so that
                # we can use simple space-separated parsing later and
                # preserve data indices.
                line = line[:17] + " " + line[18:]

                # parse data out of each line in the MODTRAN output
                toks = re.findall(r"[\S]+", line.strip())
                wl, wid = float(toks[0]), float(toks[8])  # nm
                solar_irr = float(toks[18]) * 1e6 * np.pi / wid / coszen  # uW/nm/sr/cm2
                rdnatm = float(toks[4]) * 1e6  # uW/nm/sr/cm2
                rhoatm = rdnatm * np.pi / (solar_irr * coszen)
                sphalb = float(toks[23])
                A_coeff = float(toks[21])
                B_coeff = float(toks[22])
                transm = A_coeff + B_coeff
                transup = float(toks[24])

                # Be careful with these! See note in function comments above
                thermal_emission = float(toks[11])
                thermal_scatter = float(toks[12])
                thermal_upwelling = (
                    (thermal_emission + thermal_scatter) / wid * 1e6
                )  # uW/nm/sr/cm2

                # Be careful with these! See note in function comments above
                # grnd_rflt already includes ground-to-sensor transmission
                grnd_rflt = (
                    float(toks[16]) * 1e6
                )  # ground reflected radiance (direct+diffuse+multiple scattering)
                drct_rflt = (
                    float(toks[17]) * 1e6
                )  # same as 16 but only on the sun->surface->sensor path (only direct)
                path_rdn = (
                    float(toks[14]) * 1e6 + float(toks[15]) * 1e6
                )  # The sum of the (1) single scattering and (2) multiple scattering
                thermal_downwelling = grnd_rflt / wid  # uW/nm/sr/cm2
            except:
                pdb.set_trace()

            if case[i] == 0:
                try:
                    sols.append(solar_irr)  # solar irradiance
                    transms.append(transm)  # total transmittance
                    sphalbs.append(sphalb)  # spherical albedo
                    rhoatms.append(rhoatm)  # atmospheric reflectance
                    transups.append(transup)  # upwelling direct transmittance
                    transm_dirs.append(A_coeff)  # total direct transmittance
                    transm_difs.append(B_coeff)  # total diffuse transmittance
                    widths.append(wid)  # channel width in nm
                    lp_0.append(path_rdn)  # path radiance of zero surface reflectance
                    thermal_upwellings.append(thermal_upwelling)
                    thermal_downwellings.append(thermal_downwelling)
                    wls.append(wl)  # wavelengths in nm
                except:
                    pdb.set_trace()

            elif case[i] == 1:
                try:
                    grnd_rflts_1.append(grnd_rflt)  # total ground reflected radiance
                    drct_rflts_1.append(
                        drct_rflt
                    )  # direct path ground reflected radiance
                    lp_1.append(
                        path_rdn
                    )  # path radiance (sum of single and multiple scattering)
                except:
                    pdb.set_trace()

            elif case[i] == 2:
                try:
                    grnd_rflts_2.append(grnd_rflt)  # total ground reflected radiance
                    drct_rflts_2.append(
                        drct_rflt
                    )  # direct path ground reflected radiance
                    lp_2.append(
                        path_rdn
                    )  # path radiance (sum of single and multiple scattering)
                except:
                    pdb.set_trace()

    if multipart_transmittance:
        """
        This implementation is following Gaunter et al. (2009) (DOI:10.1080/01431160802438555),
        and modified by Nimrod Carmon. It is called the "2-albedo" method, referring to running
        modtran with 2 different surface albedos. The 3-albedo method is similar to this one with
        the single difference where the "path_radiance_no_surface" variable is taken from a
        zero-surface-reflectance modtran run instead of being calculated from 2 modtran outputs.
        There are a few argument as to why this approach is beneficial:
        (1) for each grid point on the lookup table you sample modtran 2 or 3 times, i.e. you get
        2 or 3 "data points" for the atmospheric parameter of interest. This in theory allows us
        to use a lower band model resolution modtran run, which is much faster, while keeping
        high accuracy. Currently we have the 5 cm-1 band model resolution configured.
        The second advantage is the possibility to use the decoupled transmittance products to exapnd
        the forward model and account for more physics e.g. shadows \ sky view \ adjacency \ terrain etc.

        """

        t_up_dirs = np.array(transups)
        direct_ground_reflected_1 = np.array(drct_rflts_1)
        total_ground_reflected_1 = np.array(grnd_rflts_1)
        direct_ground_reflected_2 = np.array(drct_rflts_2)
        total_ground_reflected_2 = np.array(grnd_rflts_2)
        path_radiance_1 = np.array(lp_1)
        path_radiance_2 = np.array(lp_2)
        TOA_Irad = np.array(sols) * coszen / np.pi
        rfl_1 = test_rfls[1]
        rfl_2 = test_rfls[2]
        mus = coszen

        direct_flux_1 = direct_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
        global_flux_1 = total_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
        diffuse_flux_1 = global_flux_1 - direct_flux_1  # diffuse flux

        global_flux_2 = total_ground_reflected_2 * np.pi / rfl_2 / t_up_dirs

        path_radiance_no_surface = (
            rfl_2 * path_radiance_1 * global_flux_2
            - rfl_1 * path_radiance_2 * global_flux_1
        ) / (rfl_2 * global_flux_2 - rfl_1 * global_flux_1)

        # Diffuse upwelling transmittance
        t_up_difs = (
            np.pi
            * (path_radiance_1 - path_radiance_no_surface)
            / (rfl_1 * global_flux_1)
        )

        # Spherical Albedo
        sphalbs = (global_flux_1 - global_flux_2) / (
            rfl_1 * global_flux_1 - rfl_2 * global_flux_2
        )
        direct_flux_radiance = direct_flux_1 / mus

        global_flux_no_surface = global_flux_1 * (1.0 - rfl_1 * sphalbs)
        diffuse_flux_no_surface = global_flux_no_surface - direct_flux_radiance * coszen

        t_down_dirs = (direct_flux_radiance * coszen / widths / np.pi) / TOA_Irad
        t_down_difs = (diffuse_flux_no_surface / widths / np.pi) / TOA_Irad

        # total transmittance
        transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)

    if multipart_transmittance is False:
        # we need consistency in the output for later stages
        t_down_dirs = [1] * len(wls)
        t_down_difs = [1] * len(wls)
        t_up_dirs = [1] * len(wls)
        t_up_difs = [1] * len(wls)

    out_params = {
        "wls": wls,
        "sols": sols,
        "rhoatms": rhoatms,
        "transms": transms,
        "sphalbs": sphalbs,
        "t_down_dirs": t_down_dirs,
        "t_down_difs": t_down_difs,
        "t_up_dirs": t_up_dirs,
        "t_up_difs": t_up_difs,
        "thermal_upwellings": thermal_upwellings,
        "thermal_downwellings": thermal_downwellings,
    }

    return out_params
