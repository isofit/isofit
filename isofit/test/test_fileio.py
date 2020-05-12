import scipy as s
from isofit.core.fileio import typemap, max_frames_size, flush_rate


def test_typemap():
    assert typemap[s.uint64] == 15


def test_max_frames_size():
    assert max_frames_size == 100


def test_flush_rate():
    assert flush_rate == 10
