import pytest

from isofit.utils.apply_oe import apply_oe


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "requires tilt_basis_file"),
        ({"tilt_basis_file": "basis.mat"}, 'terrain_style "flat"'),
        (
            {
                "tilt_basis_file": "basis.mat",
                "terrain_style": "flat",
                "emulator_base": "emulator.h5",
            },
            "multi-part transmittance",
        ),
    ],
)
def test_tilt_coefficients_requirements(tmp_path, kwargs, match):
    """Tilt coefficients are rejected up front, before any input is read,
    without a basis, flat terrain, and multipart transmittance."""
    with pytest.raises(ValueError, match=match):
        apply_oe(
            "rdn",
            "loc",
            "obs",
            str(tmp_path / "work"),
            "emit",
            surface_path="surface.mat",
            output_tilt_coefficients_file=str(tmp_path / "tilt"),
            **kwargs,
        )
