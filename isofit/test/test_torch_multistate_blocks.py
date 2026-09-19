"""A worker's output block must preserve pixels belonging to other surface classes.

``analytical_line`` runs one worker pool per surface class
(``analytical_line.py:315``), and hands every pool line breaks spanning the whole
image (``analytical_line.py:389``). So each block a worker writes also covers
pixels that belong to *other* classes, which that worker must not disturb.

The scalar worker handles this by seeding each block from what is already on
disk (``analytical_line.py:558-585``, ``open_memmap(...)[start:stop].copy()``).
The torch worker allocated ``np.zeros`` instead, so on a multistate config the
last class to write a block erased every earlier class's pixels in it.

Single-surface runs -- the ``apply_oe`` default -- build exactly one pool, so
the bug cannot appear there. That is why it survived a 100,000-pixel end-to-end
validation. These tests use the real ENVI round trip rather than a mock, because
the invariant is about what is on disk between two passes.
"""

import numpy as np
import pytest
from spectral.io import envi

from isofit.core.common import envi_header
from isofit.core.fileio import write_bil_chunk
from isofit.utils.analytical_line_torch import read_output_block

pytestmark = pytest.mark.torch_cpu

N_LINES, N_SAMPLES, N_BANDS = 8, 4, 3


def _make_cube(tmp_path, name, fill):
    """An ENVI BIL cube on disk, prefilled with a constant."""
    path = tmp_path / name
    meta = {
        "lines": N_LINES,
        "samples": N_SAMPLES,
        "bands": N_BANDS,
        "interleave": "bil",
        "data type": 4,
        "byte order": 0,
        "header offset": 0,
    }
    envi.create_image(envi_header(str(path)), meta, ext="", force=True)
    cube = np.full((N_LINES, N_SAMPLES, N_BANDS), fill, dtype=np.float32)
    write_bil_chunk(
        np.swapaxes(cube, 1, 2), str(path), 0, (N_LINES, N_BANDS, N_SAMPLES)
    )
    return str(path)


def _read_bip(path):
    return np.array(
        envi.open(envi_header(path)).open_memmap(interleave="bip", writable=False)
    )


def test_read_output_block_returns_what_is_on_disk(tmp_path):
    """The seed must be the existing content, not zeros."""
    path = _make_cube(tmp_path, "rfl", 0.25)
    block = read_output_block(path, 2, 6)
    assert block.shape == (4, N_SAMPLES, N_BANDS)
    assert np.allclose(block, 0.25), "block was not seeded from disk"


def test_read_output_block_is_a_copy(tmp_path):
    """Mutating the block must not write through to the memmap."""
    path = _make_cube(tmp_path, "rfl", 0.25)
    block = read_output_block(path, 0, N_LINES)
    block[:] = 99.0
    assert np.allclose(_read_bip(path), 0.25), "block aliased the memmap"


def test_second_surface_class_preserves_the_first(tmp_path):
    """The regression: two classes writing the same block must both survive.

    Class A owns even lines, class B owns odd lines. Each pass covers the whole
    image, exactly as ``analytical_line`` drives it.
    """
    path = _make_cube(tmp_path, "rfl", -0.01)  # the fill value, as initialized
    shape = (N_LINES, N_BANDS, N_SAMPLES)

    for class_value, owned_rows in ((0.5, range(0, N_LINES, 2)),
                                    (0.9, range(1, N_LINES, 2))):
        block = read_output_block(path, 0, N_LINES)
        for r in owned_rows:
            block[r, :, :] = class_value
        write_bil_chunk(np.swapaxes(block, 1, 2), path, 0, shape)

    final = _read_bip(path)
    for r in range(N_LINES):
        expected = 0.5 if r % 2 == 0 else 0.9
        assert np.allclose(final[r], expected), (
            f"line {r} holds {final[r, 0, 0]}, expected {expected}: the second "
            "surface class overwrote the first"
        )


def test_zeros_seed_would_lose_the_first_class(tmp_path):
    """The same sequence seeded from zeros loses class A -- pins why this matters.

    If this ever stops failing to preserve class A, the two seeding strategies
    have become equivalent and the read-modify-write above is no longer needed.
    """
    path = _make_cube(tmp_path, "rfl", -0.01)
    shape = (N_LINES, N_BANDS, N_SAMPLES)

    for class_value, owned_rows in ((0.5, range(0, N_LINES, 2)),
                                    (0.9, range(1, N_LINES, 2))):
        block = np.zeros((N_LINES, N_SAMPLES, N_BANDS), dtype=np.float32)
        for r in owned_rows:
            block[r, :, :] = class_value
        write_bil_chunk(np.swapaxes(block, 1, 2), path, 0, shape)

    final = _read_bip(path)
    even_rows = final[0::2]
    assert np.allclose(even_rows, 0.0), (
        "expected the zeros-seeded second pass to erase the first class"
    )


def test_worker_seeds_its_block_from_disk():
    """The worker itself must use the read-modify-write, not np.zeros.

    The tests above pin the helper's semantics; this pins that ``run_chunks``
    actually calls it. Without this, reverting the worker to ``np.zeros`` leaves
    every test above passing.
    """
    import inspect

    from isofit.utils.analytical_line_torch import TorchWorker

    cls = getattr(TorchWorker, "__ray_actor_class__", TorchWorker)
    src = inspect.getsource(cls.run_chunks)
    assert "read_output_block" in src, "run_chunks does not seed its block from disk"
    assert "np.zeros" not in src.split("index_pairs")[0], (
        "run_chunks still allocates its output block from zeros"
    )
