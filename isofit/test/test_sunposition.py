from isofit.core.sunposition import _sp


def test__sp():
    sp = _sp()
    assert len(sp._EHL_) == 6
    assert len(sp._EHB_) == 2
    assert len(sp._EHR_) == 5
