from isofit.core.forward import RT_models, surface_models


def test_RT_models():
    assert len(RT_models) == 5


def test_surface_models():
    assert len(surface_models) == 7
