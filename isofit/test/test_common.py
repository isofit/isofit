import numpy as np
from isofit.core.common import max_table_size, eps, combos


def test_max_table_size():
    assert max_table_size == 500


def test_eps():
    assert eps == 1e-5


def test_combos():
    inds = np.array([[1, 2], [3, 4, 5]])
    result = np.array([[1, 3], [2, 3], [1, 4], [2, 4], [1, 5], [2, 5]])
    assert np.array_equal(combos(inds), result)
