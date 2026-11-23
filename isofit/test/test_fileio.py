from unittest.mock import MagicMock

import numpy as np

from isofit.core.fileio import (
    IO,
    InputData,
    max_frames_size,
    output_band_names,
    typemap,
)


def test_typemap():
    assert typemap[np.uint64] == 15


def test_max_frames_size():
    assert max_frames_size == 100


def test_output_band_names():
    wl_names, sv_names, tilt_names = ["Channel 0", "Channel 1"], ["AOT550"], ["PC"]

    assert output_band_names("statevector", wl_names, sv_names) == sv_names
    assert output_band_names("wavelength", wl_names, sv_names) == wl_names
    assert output_band_names("atm_coeffs", wl_names, sv_names) == wl_names * 5
    assert output_band_names("tilt_pcs", wl_names, sv_names, tilt_names) == tilt_names
    assert output_band_names("other", wl_names, sv_names) == "{}"


def _tilt_io():
    """An IO writing only tilt coefficients, without opening any files."""
    tilt_basis = MagicMock()
    tilt_basis.n_components = 3
    tilt_basis.project = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

    io = IO.__new__(IO)
    io.config = MagicMock()
    io.config.implementation.mode = "inversion"
    io.input_datasets = {}
    io.output_datasets = {"tilt_coefficients_file": None}
    io.full_statevec = ["RFL_500", "RFL_600", "AOT550"]
    io.meas_wl = np.array([500.0, 600.0])
    io.tilt_basis = tilt_basis

    fm = MagicMock()
    fm.statevec = io.full_statevec
    fm.instrument.n_chan = 2
    fm.calc_lamb = MagicMock(return_value=np.array([0.1, 0.2]))
    fm.unpack = MagicMock(
        return_value=(np.array([0.1, 0.2]), np.array([0.3]), np.zeros(0))
    )
    fm.calc_tilt_ratio = MagicMock(return_value=np.array([1.1, 0.9]))

    input_data = InputData()
    input_data.meas = np.array([5.0, 6.0])
    input_data.geom = MagicMock()

    return io, fm, input_data


def test_build_output_tilt_coefficients():
    io, fm, input_data = _tilt_io()

    to_write = io.build_output(np.array([[0.1, 0.2, 0.3]]), input_data, fm, None)

    assert list(to_write) == ["tilt_coefficients_file"]
    assert np.array_equal(to_write["tilt_coefficients_file"], [1.0, 2.0, 3.0])

    rfl, x_RT, x_instrument, geom = fm.calc_tilt_ratio.call_args.args
    assert np.array_equal(rfl, [0.1, 0.2])
    assert np.array_equal(x_RT, [0.3])
    assert geom is input_data.geom
    assert np.array_equal(io.tilt_basis.project.call_args.args[0], [1.1, 0.9])


def test_build_output_bad_data_fills_tilt():
    io, fm, input_data = _tilt_io()

    to_write = io.build_output([], input_data, fm, None)

    assert np.array_equal(to_write["tilt_coefficients_file"], np.full(3, -9999.0))
    assert len(to_write["atmospheric_coefficients_file"]) == 10
