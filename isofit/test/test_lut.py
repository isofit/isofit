"""
Much of this code is inspired by Nimrod Carmon's research 9/2023
"""
import os
import re
from glob import glob

import numpy as np
import pytest

import isofit
from isofit.utils import luts
from isofit.utils.luts import modutils


@pytest.fixture(scope="session")
def pasadena():
    """
    Loads the .chn files from the Pasadena example
    """
    files = glob(f"{isofit.root}/../examples/20171108_Pasadena/lut_multi/*.chn")
    chans = modutils.load_chns(files, True)
    datas = modutils.prepareData(chans)

    return datas


@pytest.mark.parametrize("file,group,data", [("pasadena.h5", "data", "pasadena")])
def test_combined(file, group, data, request):
    """
    Tests the write and read functions together to ensure recursive
    compatibility

    Parameters
    ----------
    request: pytest.fixture
        Built-in fixture to resolve fixtures by name

    """
    data = request.getfixturevalue(data)

    luts.writeHDF5(file, group, **data)
    read = luts.readHDF5(file, group)

    # Cleanup, don't need anymore
    os.remove(file)

    for key in data:
        dtype = read[key].dtype
        assert (read[key] == np.array(data[key])).all()
