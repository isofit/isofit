"""
Tests for isofit.utils.neon_h5

A synthetic NEON HDF5 file is constructed in a temporary directory for each
test so no real data file is required.  The structure mirrors the actual NEON
L1 radiance product layout.
"""

import importlib.util
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module directly to avoid pulling in the full isofit package,
# which requires optional heavy dependencies (torch, etc.).
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "neon_h5",
    Path(__file__).resolve().parents[1] / "utils" / "neon_h5.py",
)
neon_h5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(neon_h5)

convert_rad = neon_h5.convert_rad
convert_loc = neon_h5.convert_loc
convert_obs = neon_h5.convert_obs
convert_neon_h5 = neon_h5.convert_neon_h5
LOC_BAND_NAMES = neon_h5.LOC_BAND_NAMES
OBS_BAND_NAMES = neon_h5.OBS_BAND_NAMES
_to_decimal_hours = neon_h5._to_decimal_hours
_get_scale = neon_h5._get_scale

# ---------------------------------------------------------------------------
# Synthetic H5 fixture
# ---------------------------------------------------------------------------

SITE = "NEON_TEST"
N_LINES, N_SAMPLES, N_BANDS = 4, 5, 6
SCALE = 10_000.0

WAVELENGTHS = np.array([400.0, 420.0, 440.0, 460.0, 480.0, 500.0], dtype=np.float32)
FWHM = np.full(N_BANDS, 10.0, dtype=np.float32)
MAP_INFO = "UTM, 1, 1, 480000.0, 4442000.0, 1.0, 1.0, 13, North, WGS-84, units=Meters"
COORD_SYS = "PROJCS[WGS 84 / UTM zone 13N]"


def _make_h5(path: Path) -> Path:
    """Write a minimal synthetic NEON H5 file to *path* and return it."""
    rng = np.random.default_rng(0)

    int_part = rng.integers(50, 200, size=(N_LINES, N_SAMPLES, N_BANDS), dtype=np.int16)
    dec_part = rng.integers(0, 9999, size=(N_LINES, N_SAMPLES, N_BANDS), dtype=np.int16)

    igm = rng.uniform(
        [480000, 4_442_000, 1600],
        [480100, 4_442_100, 1700],
        size=(N_LINES, N_SAMPLES, 3),
    ).astype(np.float32)

    obs = rng.uniform(0, 1, size=(N_LINES, N_SAMPLES, 10)).astype(np.float32)
    obs[:, :, 9] = 58_500.0  # seconds since midnight → 16.25 decimal hours

    with h5py.File(path, "w") as f:
        rad = f.create_group(f"{SITE}/Radiance")

        ds_int = rad.create_dataset("RadianceIntegerPart", data=int_part)
        ds_int.attrs["Data_Ignore_Value"] = -9999

        ds_dec = rad.create_dataset("RadianceDecimalPart", data=dec_part)
        ds_dec.attrs["Data_Ignore_Value"] = -9999
        ds_dec.attrs["Scale_Factor"] = SCALE

        spec = rad.create_group("Metadata/Spectral_Data")
        spec.create_dataset("Wavelength", data=WAVELENGTHS)
        spec.create_dataset("FWHM", data=FWHM)

        crs = rad.create_group("Metadata/Coordinate_System")
        crs.create_dataset("Map_Info", data=MAP_INFO.encode())
        crs.create_dataset("Coordinate_System_String", data=COORD_SYS.encode())

        anc = rad.create_group("Metadata/Ancillary_Rasters")
        anc.create_dataset("IGM_Data", data=igm)
        anc.create_dataset("OBS_Data", data=obs)

    return path


@pytest.fixture
def h5_file(tmp_path):
    return _make_h5(tmp_path / "test_flight.h5")


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_to_decimal_hours_passthrough():
    """Values already in decimal hours should pass through unchanged."""
    arr = np.array([16.25, 0.0, 23.99], dtype=np.float32)
    result = _to_decimal_hours(arr)
    np.testing.assert_allclose(result, arr, rtol=1e-5)


def test_to_decimal_hours_from_seconds():
    """Seconds since midnight should convert to decimal hours."""
    arr = np.array([58_500.0], dtype=np.float32)  # 16.25 h × 3600
    result = _to_decimal_hours(arr)
    np.testing.assert_allclose(result, [16.25], rtol=1e-4)


def test_to_decimal_hours_from_hhmmss():
    """HHMMSS numeric format (e.g. 162500 = 16:25:00) should convert correctly."""
    arr = np.array([162_500.0], dtype=np.float64)
    result = _to_decimal_hours(arr)
    np.testing.assert_allclose(result, [16.0 + 25.0 / 60.0], rtol=1e-4)


def test_get_scale_reads_attribute(h5_file):
    """_get_scale should find Scale_Factor on the decimal-part dataset."""
    with h5py.File(h5_file, "r") as f:
        rad = f[SITE]["Radiance"]
        scale = _get_scale(rad["RadianceDecimalPart"], rad)
    assert scale == SCALE


# ---------------------------------------------------------------------------
# convert_rad
# ---------------------------------------------------------------------------


def test_convert_rad_files_created(tmp_path, h5_file):
    base = str(tmp_path / "out")
    rad_path, hdr_path = convert_rad(str(h5_file), base)
    assert Path(rad_path).exists()
    assert Path(hdr_path).exists()


def test_convert_rad_file_size(tmp_path, h5_file):
    """BIL float32: file size = lines × bands × samples × 4 bytes."""
    base = str(tmp_path / "out")
    rad_path, _ = convert_rad(str(h5_file), base)
    expected = N_LINES * N_BANDS * N_SAMPLES * 4
    assert Path(rad_path).stat().st_size == expected


def test_convert_rad_values(tmp_path, h5_file):
    """Reconstructed radiance should equal integer + decimal / scale."""
    base = str(tmp_path / "out")
    convert_rad(str(h5_file), base)

    # Read back the BIL binary manually
    raw = np.fromfile(base + ".rad", dtype=np.float32)
    # BIL layout: (lines, bands, samples) → reshape and transpose to (lines, samples, bands)
    bil = raw.reshape(N_LINES, N_BANDS, N_SAMPLES).transpose(0, 2, 1)

    with h5py.File(h5_file, "r") as f:
        rad = f[SITE]["Radiance"]
        int_part = rad["RadianceIntegerPart"][:].astype(np.float32)
        dec_part = rad["RadianceDecimalPart"][:].astype(np.float32)

    expected = int_part + dec_part / SCALE
    np.testing.assert_allclose(bil, expected, rtol=1e-5)


def test_convert_rad_header_content(tmp_path, h5_file):
    base = str(tmp_path / "out")
    _, hdr_path = convert_rad(str(h5_file), base)
    hdr = Path(hdr_path).read_text()
    assert "interleave = bil" in hdr
    assert "data type = 4" in hdr
    assert "400.000000" in hdr  # first wavelength


# ---------------------------------------------------------------------------
# convert_loc
# ---------------------------------------------------------------------------


def test_convert_loc_files_created(tmp_path, h5_file):
    base = str(tmp_path / "out")
    loc_path, hdr_path = convert_loc(str(h5_file), base)
    assert Path(loc_path).exists()
    assert Path(hdr_path).exists()


def test_convert_loc_file_size(tmp_path, h5_file):
    """BIL float32: lines × 3 bands × samples × 4 bytes."""
    base = str(tmp_path / "out")
    loc_path, _ = convert_loc(str(h5_file), base)
    expected = N_LINES * 3 * N_SAMPLES * 4
    assert Path(loc_path).stat().st_size == expected


def test_convert_loc_band_names_in_header(tmp_path, h5_file):
    base = str(tmp_path / "out")
    _, hdr_path = convert_loc(str(h5_file), base)
    hdr = Path(hdr_path).read_text()
    for name in LOC_BAND_NAMES:
        assert name in hdr


def test_convert_loc_values(tmp_path, h5_file):
    """LOC values should match the IGM_Data stored in the H5 file."""
    base = str(tmp_path / "out")
    loc_path, _ = convert_loc(str(h5_file), base)

    raw = np.fromfile(loc_path, dtype=np.float32)
    bil = raw.reshape(N_LINES, 3, N_SAMPLES).transpose(0, 2, 1)

    with h5py.File(h5_file, "r") as f:
        igm = f[SITE]["Radiance"]["Metadata"]["Ancillary_Rasters"]["IGM_Data"][:]

    np.testing.assert_allclose(bil, igm, rtol=1e-5)


# ---------------------------------------------------------------------------
# convert_obs
# ---------------------------------------------------------------------------


def test_convert_obs_files_created(tmp_path, h5_file):
    base = str(tmp_path / "out")
    obs_path, hdr_path = convert_obs(str(h5_file), base)
    assert Path(obs_path).exists()
    assert Path(hdr_path).exists()


def test_convert_obs_file_size(tmp_path, h5_file):
    """BIP float32: lines × samples × 10 bands × 4 bytes."""
    base = str(tmp_path / "out")
    obs_path, _ = convert_obs(str(h5_file), base)
    expected = N_LINES * N_SAMPLES * 10 * 4
    assert Path(obs_path).stat().st_size == expected


def test_convert_obs_interleave_is_bip(tmp_path, h5_file):
    base = str(tmp_path / "out")
    _, hdr_path = convert_obs(str(h5_file), base)
    assert "interleave = bip" in Path(hdr_path).read_text()


def test_convert_obs_band_names_in_header(tmp_path, h5_file):
    base = str(tmp_path / "out")
    _, hdr_path = convert_obs(str(h5_file), base)
    hdr = Path(hdr_path).read_text()
    for name in OBS_BAND_NAMES:
        assert name in hdr


def test_convert_obs_time_converted(tmp_path, h5_file):
    """Band 10 (index 9) should be converted from seconds to decimal hours."""
    base = str(tmp_path / "out")
    obs_path, _ = convert_obs(str(h5_file), base)

    # BIP layout: (lines, samples, bands) in C order
    raw = np.fromfile(obs_path, dtype=np.float32).reshape(N_LINES, N_SAMPLES, 10)
    time_band = raw[:, :, 9]

    # 58500 seconds → 16.25 hours
    np.testing.assert_allclose(time_band, 16.25, rtol=1e-3)


# ---------------------------------------------------------------------------
# convert_neon_h5 (end-to-end triplet)
# ---------------------------------------------------------------------------


def test_convert_neon_h5_all_files_created(tmp_path, h5_file):
    paths = convert_neon_h5(str(h5_file), str(tmp_path))
    for key in ("rad", "rad_hdr", "loc", "loc_hdr", "obs", "obs_hdr"):
        assert key in paths
        assert Path(paths[key]).exists(), f"Missing: {key}"


def test_convert_neon_h5_subdir(tmp_path, h5_file):
    """With subdir=True (default) files should land in <out>/<stem>/."""
    paths = convert_neon_h5(str(h5_file), str(tmp_path))
    stem = Path(h5_file).stem
    assert Path(paths["rad"]).parent.name == stem


def test_convert_neon_h5_no_subdir(tmp_path, h5_file):
    """With subdir=False files should land directly in out_root."""
    paths = convert_neon_h5(str(h5_file), str(tmp_path), subdir=False)
    assert Path(paths["rad"]).parent == tmp_path
