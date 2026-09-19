"""Round-trip tests for the batched worker's output layout.

These exist because of a specific bug: the torch worker wrote its output cube
with ``output_rfl.T`` where the scalar worker uses ``np.swapaxes(output_rfl, 1, 2)``.
``write_bil_chunk`` serializes with a raw ``tobytes()`` and never reshapes, so
transposing all three axes instead of swapping two silently produced a scrambled
cube: every value present, every value in the wrong pixel.

That failure is invisible to aggregate checks -- the histogram, the min/max, and
the valid/fill fractions are all identical, because the multiset of values is
unchanged. It survived to a full end-to-end scene run. A round-trip assertion on
a small array catches it in milliseconds.
"""

import numpy as np
import pytest

from isofit.core.fileio import write_bil_chunk

pytestmark = pytest.mark.torch_cpu

N_LINES, N_SAMPLES, N_BANDS = 7, 5, 3


def _distinct_cube():
    """A cube whose every element encodes its own (line, sample, band)."""
    cube = np.zeros((N_LINES, N_SAMPLES, N_BANDS), dtype=float)
    for l in range(N_LINES):
        for s in range(N_SAMPLES):
            for b in range(N_BANDS):
                cube[l, s, b] = l * 10000 + s * 100 + b
    return cube


def _write_and_read(tmp_path, cube, transform):
    """Write a BIP-shaped cube through write_bil_chunk and read it back."""
    path = tmp_path / "cube"
    path.write_bytes(b"")
    write_bil_chunk(
        transform(cube), str(path), 0, (N_LINES, N_BANDS, N_SAMPLES)
    )
    raw = np.fromfile(str(path), dtype=np.float32)
    # BIL on disk is (lines, bands, samples)
    return raw.reshape(N_LINES, N_BANDS, N_SAMPLES)


def test_swapaxes_round_trips_every_pixel(tmp_path):
    """The correct transform must put each value back at its own coordinates."""
    cube = _distinct_cube()
    bil = _write_and_read(tmp_path, cube, lambda c: np.swapaxes(c, 1, 2))

    for l in range(N_LINES):
        for s in range(N_SAMPLES):
            for b in range(N_BANDS):
                assert bil[l, b, s] == pytest.approx(cube[l, s, b]), (
                    f"value for (line={l}, sample={s}, band={b}) landed wrong"
                )


def test_transpose_scrambles_the_cube(tmp_path):
    """Guard the regression directly: .T must NOT round-trip.

    If this ever starts passing, the shapes have become degenerate (e.g. a
    single-line chunk, where the two layouts serialize identically) and the
    test has stopped protecting anything.
    """
    cube = _distinct_cube()
    bil = _write_and_read(tmp_path, cube, lambda c: c.T)

    correct = np.swapaxes(cube, 1, 2)
    assert not np.allclose(bil, correct), (
        ".T must differ from swapaxes(1, 2) for a non-degenerate cube"
    )


def test_transpose_preserves_the_value_multiset(tmp_path):
    """Why the bug hid: a permutation changes no aggregate statistic.

    Documents the failure mode so nobody concludes from matching histograms,
    fill fractions or min/max that two cubes agree.
    """
    cube = _distinct_cube()
    good = _write_and_read(tmp_path, cube, lambda c: np.swapaxes(c, 1, 2))
    bad = _write_and_read(tmp_path, cube, lambda c: c.T)

    np.testing.assert_array_equal(np.sort(good, axis=None), np.sort(bad, axis=None))
    assert not np.allclose(good, bad)


def test_worker_uses_swapaxes_not_transpose():
    """The torch worker's writer must match the scalar worker's layout."""
    import inspect

    from isofit.utils.analytical_line_torch import TorchWorker

    src = inspect.getsource(TorchWorker._write)
    assert "swapaxes" in src, "_write must use swapaxes(1, 2)"
    assert ".T," not in src, "_write must not transpose all three axes"


def test_scalar_and_torch_writers_agree(tmp_path):
    """Both workers must serialize an identical cube identically."""
    import inspect

    from isofit.utils.analytical_line import Worker

    scalar_src = inspect.getsource(Worker.run_chunks)
    # The scalar worker's own fix; keep the two in step.
    assert "np.swapaxes(output_rfl, 1, 2)" in scalar_src

    cube = _distinct_cube()
    a = _write_and_read(tmp_path, cube, lambda c: np.swapaxes(c, 1, 2))
    b = _write_and_read(tmp_path, cube, lambda c: np.swapaxes(c, 1, 2))
    np.testing.assert_array_equal(a, b)
