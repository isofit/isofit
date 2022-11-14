import scipy as s

from isofit.core.fileio import max_frames_size, typemap


def test_typemap():
    assert typemap[s.uint64] == 15


def test_max_frames_size():
    assert max_frames_size == 100
