import os
import shutil
from functools import partial
from glob import glob
from pathlib import Path

import ray

from isofit.utils.luts import modutils


def sim(command, output):
    """
    Fake simulation function. In other words, simulates simulations.
    """
    output = os.path.join(output, os.path.basename(command))
    shutil.copy(command, output)

    return str(output)


def writeNetCDF(file, data):
    """
    To be developed, similar to luts.writeHDF5
    """
    data.to_netcdf(file)


@ray.remote
def runSim(command, output, reader, tmp=None):
    """
    Parallelizable function
    """
    # Step 1: Run simulation
    file = sim(command, output)

    # Step 2: Read results
    data = reader(file)

    # Step 3: Write to NetCDF

    if tmp is None:  # Save to independent .nc file for easier merging afterwards
        tmp = ray.get_tmp_dir()

    writeNetCDF(f"{tmp}/{os.path.basename(file)}", data=data)


class RTE:
    def __init__(self, name, output):
        """
        Generalized RTE class
        """
        # Create output directories if they don't exist
        self.output = Path(output) / name
        self.lfile = self.output / "luts.nc"

        self.sims = self.output / "sims"
        self.sims.mkdir(mode=0o777, parents=True, exist_ok=True)

        self.tmp = self.output / "tmp"
        self.tmp.mkdir(mode=0o777, parents=True, exist_ok=True)

    def reader(self, *args, **kwargs):
        """
        Reader function for simulation results
        """
        raise NotImplementedError(
            f"RTE subclass missing reader function: {self.__class__.__name__}"
        )

    def runSims(self, commands):
        """ """
        runner = partial(
            runSim.remote, output=str(self.sims), reader=self.reader, tmp=str(self.tmp)
        )
        # Process all sims
        ray.get([runner(cmd) for cmd in commands])

        # Merge sims
        ds = xr.open_mfdataset(f"{self.tmp}/*", combine="nested", concat_dim="lut")

        ds.to_netcdf(self.lfile)

        # Clean up the individual LUT.nc files
        shutil.rmtree(str(self.tmp))

        return ds


class MODTRAN(RTE):
    """
    RTE subclass
    """

    def __init__(self, output):
        super().__init__(self.__class__.__name__, output)

    def reader(self, file, **kwargs):
        return modutils.parseChannelXarray(file, **kwargs)


#%%

files = glob("examples/20171108_Pasadena/lut_multi/*.chn")
rt = MODTRAN(output="isofit/utils/luts/.idea")

ds = rt.runSims(files)
