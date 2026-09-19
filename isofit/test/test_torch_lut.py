"""Parity tests for the batched torch LUT interpolator.

The reference is ``isofit.core.common.VectorInterpolator`` in its default
``mlg_numba`` mode -- the exact kernel the numpy retrieval path uses. Agreement
must hold not just on random interior points but on the adversarial coordinates
where the two implementations could plausibly diverge: exact grid nodes (where
``searchsorted`` picks a side), points below/above the grid (clamping), and
values sitting exactly on cell boundaries.

Small interpolation errors are not benign here: the LUT feeds water-vapor and
aerosol retrieval, so a wrong cell selection is a discontinuous jump, not a
rounding difference.
"""

import numpy as np
import pytest
import torch

from isofit.backends.torch.lut import BatchedLUT
from isofit.core.common import VectorInterpolator

FP64_RTOL = 1e-13
FP32_RTOL = 3e-6

# Every test in this module exercises the torch backend's numerics on CPU.
pytestmark = pytest.mark.torch_cpu


def _accelerator():
    """Return the CUDA device, or None.

    CUDA is the only supported accelerator. MPS is deliberately excluded: it has
    no float64, so it cannot validate a backend whose linear algebra is float64.
    """
    return torch.device("cuda") if torch.cuda.is_available() else None


requires_gpu = pytest.mark.skipif(
    _accelerator() is None, reason="requires a CUDA device"
)


# --- fixtures -------------------------------------------------------------------


def _make_lut(shape=(5, 4, 3), n_wl=12, seed=0, irregular=True):
    """Build a random LUT with the given grid shape and channel count."""
    rng = np.random.default_rng(seed)

    grids = []
    for n in shape:
        if n == 1:
            grids.append(np.array([0.5]))
        elif irregular:
            # Non-uniform spacing: exercises the per-cell width division.
            g = np.sort(rng.uniform(0, 1, n))
            # Guarantee separation so the division is well conditioned
            g = np.linspace(0, 1, n) + 0.03 * (g - 0.5)
            grids.append(np.sort(g))
        else:
            grids.append(np.linspace(0.0, 1.0, n))

    data = rng.normal(size=(*shape, n_wl))
    return grids, data


def _reference(grids, data, points):
    """Interpolate each point with the numba reference kernel."""
    itp = VectorInterpolator(list(grids), data, version="mlg_numba")
    return np.array([itp(np.asarray(p, dtype=np.float64)) for p in points])


def _batched(grids, data, points, dtype=torch.float64, device="cpu"):
    lut = BatchedLUT(grids, {"q": data}, device=device, dtype=dtype)
    return lut.interpolate(torch.as_tensor(np.asarray(points)))["q"]


def _assert_parity(grids, data, points, rtol=FP64_RTOL, dtype=torch.float64):
    ref = _reference(grids, data, points)
    got = _batched(grids, data, points, dtype=dtype).cpu().numpy()
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=rtol)


# --- interior points ------------------------------------------------------------


def test_random_interior_points_match():
    grids, data = _make_lut()
    rng = np.random.default_rng(1)
    points = np.stack(
        [rng.uniform(g[0], g[-1], 200) for g in grids], axis=1
    )
    _assert_parity(grids, data, points)


def test_uniform_grid_matches():
    grids, data = _make_lut(irregular=False)
    rng = np.random.default_rng(2)
    points = np.stack([rng.uniform(g[0], g[-1], 100) for g in grids], axis=1)
    _assert_parity(grids, data, points)


# --- adversarial coordinates ----------------------------------------------------


def _node_points_and_values(grids, data):
    """Every grid node, paired with the table value that sits on it.

    Multilinear interpolation evaluated exactly at a node must return that
    node's value. That is an analytic ground truth, so node tests assert against
    it directly rather than against another implementation.
    """
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=1)
    values = data.reshape(-1, data.shape[-1])
    return points, values


def test_exact_grid_nodes_return_node_values():
    """Exact node hits must reproduce the table exactly."""
    grids, data = _make_lut()
    points, values = _node_points_and_values(grids, data)
    got = _batched(grids, data, points).cpu().numpy()
    np.testing.assert_allclose(got, values, rtol=FP64_RTOL, atol=FP64_RTOL)


def test_exact_grid_nodes_match_reference():
    """The numba reference should agree at nodes too.

    Kept separate from the analytic check above because it compares two
    implementations rather than establishing correctness. See
    ``test_exact_grid_nodes_return_node_values`` for the authoritative
    assertion.
    """
    grids, data = _make_lut()
    points, _ = _node_points_and_values(grids, data)
    _assert_parity(grids, data, points)


def test_points_below_grid_are_clamped_like_reference():
    grids, data = _make_lut()
    points = np.stack(
        [np.full(20, g[0] - 5.0) for g in grids], axis=1
    )
    _assert_parity(grids, data, points)


def test_points_above_grid_are_clamped_like_reference():
    grids, data = _make_lut()
    points = np.stack([np.full(20, g[-1] + 5.0) for g in grids], axis=1)
    _assert_parity(grids, data, points)


def test_mixed_inside_and_outside_dimensions():
    """One axis clamped low, one clamped high, one interior."""
    grids, data = _make_lut(shape=(5, 4, 3))
    points = np.array(
        [
            [grids[0][0] - 1.0, grids[1][-1] + 1.0, 0.5],
            [grids[0][-1] + 1.0, grids[1][0] - 1.0, 0.25],
            [grids[0][0], grids[1][-1], grids[2][0]],
        ]
    )
    _assert_parity(grids, data, points)


def test_cell_midpoints_match():
    """Exactly halfway between nodes -- equal corner weights."""
    grids, data = _make_lut()
    mids = [(g[:-1] + g[1:]) / 2 for g in grids]
    n = min(len(m) for m in mids)
    points = np.stack([m[:n] for m in mids], axis=1)
    _assert_parity(grids, data, points)


def test_epsilon_around_nodes_match():
    """Just below / just above an interior node, where cell choice flips."""
    grids, data = _make_lut()
    node = np.array([g[1] for g in grids])
    eps = 1e-12
    points = np.stack([node - eps, node, node + eps])
    _assert_parity(grids, data, points)


# --- shapes and dimensionality --------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (3, 5), (3, 4, 2), (2, 3, 2, 3), (2, 2, 2, 2, 2)])
def test_various_dimensionalities(shape):
    """ISOFIT LUTs run 3-5 dimensions; cover the surrounding range."""
    grids, data = _make_lut(shape=shape, n_wl=7, seed=len(shape))
    rng = np.random.default_rng(len(shape))
    points = np.stack([rng.uniform(g[0], g[-1], 50) for g in grids], axis=1)
    _assert_parity(grids, data, points)


def test_singleton_dimension_matches():
    """A collapsed grid axis must not walk off the end of the table."""
    grids, data = _make_lut(shape=(4, 1, 3))
    rng = np.random.default_rng(7)
    points = np.stack(
        [
            rng.uniform(grids[0][0], grids[0][-1], 30),
            rng.uniform(-1.0, 2.0, 30),  # spans the singleton node
            rng.uniform(grids[2][0], grids[2][-1], 30),
        ],
        axis=1,
    )
    _assert_parity(grids, data, points)


def test_gather_indices_stay_in_bounds_with_singleton_dim():
    """Directly assert no corner gather can run off the end of the table.

    End-to-end value parity alone does not cover this: a zero-weighted corner
    can hide an out-of-range index. Recomputing the indices the way
    interpolate_stacked does keeps the guarantee explicit.
    """
    grids, data = _make_lut(shape=(4, 1, 3))
    lut = BatchedLUT(grids, {"q": data})
    n_cells = int(np.prod(lut.shape))

    rng = np.random.default_rng(21)
    points = torch.as_tensor(
        np.stack(
            [
                rng.uniform(-1, 2, 200),
                rng.uniform(-1, 2, 200),
                rng.uniform(-1, 2, 200),
            ],
            axis=1,
        )
    )

    low, _ = lut._weights_and_cells(points)
    upper = low + 1

    for corner in lut.corners:
        cell = torch.zeros(points.shape[0], dtype=torch.int64)
        for d in range(lut.dims):
            src = upper if (corner >> d) & 1 else low
            cell = cell + src[:, d] * lut.strides[d]
        assert int(cell.min()) >= 0, f"corner {corner} produced a negative index"
        assert int(cell.max()) < n_cells, f"corner {corner} ran past the table"


def test_degenerate_axis_corners_are_dropped_statically():
    """Corners stepping up a length-1 axis contribute zero and must not be run."""
    grids, data = _make_lut(shape=(4, 1, 3))
    lut = BatchedLUT(grids, {"q": data})
    assert len(lut.corners) == 4, "expected 2**2 live corners for one degenerate axis"
    assert all(not ((c >> 1) & 1) for c in lut.corners)


def test_all_corners_live_without_degenerate_axes():
    grids, data = _make_lut(shape=(4, 3, 3))
    lut = BatchedLUT(grids, {"q": data})
    assert len(lut.corners) == 8


def test_single_point_input_is_accepted():
    """A bare 1-D coordinate is promoted to a batch of one."""
    grids, data = _make_lut()
    lut = BatchedLUT(grids, {"q": data})
    point = np.array([0.3, 0.4, 0.5])
    out = lut.interpolate(torch.as_tensor(point))["q"]
    assert out.shape == (1, data.shape[-1])
    ref = _reference(grids, data, [point])
    np.testing.assert_allclose(out.numpy(), ref, rtol=FP64_RTOL)


def test_large_batch_matches():
    grids, data = _make_lut(shape=(6, 5, 4), n_wl=20)
    rng = np.random.default_rng(11)
    points = np.stack([rng.uniform(-0.2, 1.2, 5000) for g in grids], axis=1)
    _assert_parity(grids, data, points)


# --- multi-quantity fusion ------------------------------------------------------


def test_multiple_quantities_match_individual_interpolation():
    """The fused stacked layout must equal per-quantity interpolation."""
    grids, _ = _make_lut()
    rng = np.random.default_rng(3)
    shape = tuple(len(g) for g in grids)
    quantities = {f"q{i}": rng.normal(size=(*shape, 9)) for i in range(5)}

    points = np.stack([rng.uniform(g[0], g[-1], 64) for g in grids], axis=1)

    lut = BatchedLUT(grids, quantities)
    got = lut.interpolate(torch.as_tensor(points))

    for key, arr in quantities.items():
        ref = _reference(grids, arr, points)
        np.testing.assert_allclose(got[key].numpy(), ref, rtol=FP64_RTOL)


def test_constants_passed_through():
    """Constant quantities mirror VectorInterpolator's method == -1 shortcut."""
    grids, data = _make_lut()
    lut = BatchedLUT(grids, {"q": data}, constants={"sphalb": 0.125})
    out = lut.interpolate(torch.as_tensor(np.array([[0.5, 0.5, 0.5]])))
    assert out["sphalb"] == 0.125
    assert out["q"].shape == (1, data.shape[-1])


def test_from_interpolators_bridges_reader_output():
    """Construction from VectorInterpolator objects, as the LUT Reader builds them."""
    grids, data = _make_lut()
    other = np.asarray(np.random.default_rng(5).normal(size=data.shape))

    interpolators = {
        "rhoatm": VectorInterpolator(list(grids), data, version="mlg_numba"),
        "sphalb": VectorInterpolator(list(grids), other, version="mlg_numba"),
    }
    lut = BatchedLUT.from_interpolators(interpolators)

    rng = np.random.default_rng(6)
    points = np.stack([rng.uniform(g[0], g[-1], 32) for g in grids], axis=1)
    got = lut.interpolate(torch.as_tensor(points))

    np.testing.assert_allclose(
        got["rhoatm"].numpy(), _reference(grids, data, points), rtol=FP64_RTOL
    )
    np.testing.assert_allclose(
        got["sphalb"].numpy(), _reference(grids, other, points), rtol=FP64_RTOL
    )


def test_from_interpolators_carries_constant_quantities():
    grids, data = _make_lut()
    const = np.full((*tuple(len(g) for g in grids), 12), 0.25)

    interpolators = {
        "rhoatm": VectorInterpolator(list(grids), data, version="mlg_numba"),
        "flat": VectorInterpolator(list(grids), const, version="mlg_numba"),
    }
    assert interpolators["flat"].method == -1, "expected the constant shortcut"

    lut = BatchedLUT.from_interpolators(interpolators)
    out = lut.interpolate(torch.as_tensor(np.array([[0.5, 0.5, 0.5]])))
    assert out["flat"] == pytest.approx(0.25)


# --- analytic gradients ---------------------------------------------------------
#
# The reference is a CENTRAL difference of the interpolator itself, not of the
# numba kernel: the claim under test is that interpolate_with_gradients returns
# the derivative of the very interpolant BatchedLUT implements, clamping and
# all. Inside a cell the multilinear form is exactly linear in each coordinate,
# so a central difference taken away from the cell walls is exact up to
# cancellation -- which is why the tolerances below are tight.


GRAD_RTOL = 1e-8
GRAD_ATOL = 1e-8


def _interior_points(grids, n, rng, lo=0.25, hi=0.75):
    """Random points sitting well inside a cell, away from every wall."""
    cols = []
    for g in grids:
        if len(g) == 1:
            cols.append(np.full(n, g[0]))
            continue
        cell = rng.integers(0, len(g) - 1, n)
        frac = rng.uniform(lo, hi, n)
        cols.append(g[cell] + frac * (g[cell + 1] - g[cell]))
    return np.stack(cols, axis=1)


def _central_difference(lut, points, d, key="q", h=1e-6):
    """d/d p_d of the interpolator, by central difference of the interpolator."""
    step = np.zeros_like(points)
    step[:, d] = h
    plus = lut.interpolate(torch.as_tensor(points + step))[key]
    minus = lut.interpolate(torch.as_tensor(points - step))[key]
    return ((plus - minus) / (2 * h)).cpu().numpy()


def _assert_gradients_match_central_difference(lut, points, key="q"):
    """Every axis's analytic gradient must match a central difference."""
    _, grads = lut.interpolate_with_gradients(torch.as_tensor(points))
    for d in range(lut.dims):
        ref = _central_difference(lut, points, d, key=key)
        np.testing.assert_allclose(
            grads[key][:, d].cpu().numpy(), ref, rtol=GRAD_RTOL, atol=GRAD_ATOL
        )


def test_gradients_match_central_difference():
    """Analytic derivative vs a central difference, on interior points."""
    grids, data = _make_lut(shape=(5, 4, 3), n_wl=12)
    lut = BatchedLUT(grids, {"q": data})

    rng = np.random.default_rng(101)
    _assert_gradients_match_central_difference(
        lut, _interior_points(grids, 200, rng)
    )


@pytest.mark.parametrize("shape", [(4,), (3, 5), (3, 4, 2), (2, 3, 2, 3)])
def test_gradients_match_central_difference_various_dimensionalities(shape):
    grids, data = _make_lut(shape=shape, n_wl=7, seed=len(shape))
    lut = BatchedLUT(grids, {"q": data})

    rng = np.random.default_rng(len(shape))
    _assert_gradients_match_central_difference(
        lut, _interior_points(grids, 64, rng)
    )


def test_gradient_values_match_plain_interpolation():
    """The fused call must not perturb the values it returns alongside."""
    grids, data = _make_lut()
    lut = BatchedLUT(grids, {"q": data})

    rng = np.random.default_rng(102)
    points = torch.as_tensor(_interior_points(grids, 128, rng))

    values, _ = lut.interpolate_with_gradients(points)
    np.testing.assert_array_equal(
        values["q"].numpy(), lut.interpolate(points)["q"].numpy()
    )


def test_linear_ramp_has_exactly_constant_gradient():
    """A table that is linear in the grid coordinates has an analytic answer.

    Building the table as ``sum_d slope_d * p_d`` makes the multilinear
    interpolant exact everywhere, so the derivative is the slope -- a ground
    truth that owes nothing to another implementation.
    """
    grids = [
        np.linspace(0.0, 2.0, 5),
        np.linspace(-1.0, 1.0, 4),
        np.linspace(10.0, 14.0, 3),
    ]
    slopes = np.array([2.0, -3.0, 0.5])

    mesh = np.meshgrid(*grids, indexing="ij")
    ramp = sum(s * m for s, m in zip(slopes, mesh))
    # Per-channel offsets and scalings, so the test is not accidentally passing
    # on a single degenerate channel.
    channel = np.arange(1, 7, dtype=float)
    data = ramp[..., None] * channel + channel

    lut = BatchedLUT(grids, {"q": data})

    rng = np.random.default_rng(103)
    points = np.stack([rng.uniform(g[0] + 0.1, g[-1] - 0.1, 50) for g in grids], axis=1)

    values, grads = lut.interpolate_with_gradients(torch.as_tensor(points))

    expected_values = (points @ slopes)[:, None] * channel + channel
    np.testing.assert_allclose(values["q"].numpy(), expected_values, rtol=FP64_RTOL)

    for d, slope in enumerate(slopes):
        expected = np.broadcast_to(slope * channel, (len(points), len(channel)))
        np.testing.assert_allclose(
            grads["q"][:, d].numpy(), expected, rtol=FP64_RTOL, atol=FP64_RTOL
        )


def test_gradient_is_zero_outside_the_grid():
    """Clamped coordinates make the interpolant locally constant."""
    grids, data = _make_lut()
    lut = BatchedLUT(grids, {"q": data})

    below = np.stack([np.full(10, g[0] - 5.0) for g in grids], axis=1)
    above = np.stack([np.full(10, g[-1] + 5.0) for g in grids], axis=1)

    for points in (below, above):
        _, grads = lut.interpolate_with_gradients(torch.as_tensor(points))
        assert torch.count_nonzero(grads["q"]) == 0


def test_gradient_is_zero_only_on_the_clamped_axis():
    """A point outside along one axis still has a live derivative on the others."""
    grids, data = _make_lut(shape=(5, 4, 3), n_wl=12)
    lut = BatchedLUT(grids, {"q": data})

    rng = np.random.default_rng(104)
    points = _interior_points(grids, 32, rng)
    points[:, 1] = grids[1][-1] + 3.0  # clamped high on axis 1 only

    _, grads = lut.interpolate_with_gradients(torch.as_tensor(points))

    assert torch.count_nonzero(grads["q"][:, 1]) == 0
    for d in (0, 2):
        ref = _central_difference(lut, points, d)
        np.testing.assert_allclose(
            grads["q"][:, d].numpy(), ref, rtol=GRAD_RTOL, atol=GRAD_ATOL
        )
        assert np.abs(ref).max() > 0, "expected a live derivative on this axis"


def test_gradient_along_degenerate_axis_is_zero():
    """A length-1 axis contributes nothing, so its derivative is exactly zero.

    The corner sum cannot produce this on its own: ``self.corners`` drops the
    degenerate axis's up-corners, so a naive signed sum would return the
    negated value rather than zero.
    """
    grids, data = _make_lut(shape=(4, 1, 3))
    lut = BatchedLUT(grids, {"q": data})

    rng = np.random.default_rng(105)
    points = _interior_points(grids, 40, rng)
    # Sweep the collapsed axis well past its single node in both directions.
    points[:, 1] = rng.uniform(-2.0, 3.0, len(points))

    _, grads = lut.interpolate_with_gradients(torch.as_tensor(points))

    assert torch.count_nonzero(grads["q"][:, 1]) == 0
    for d in (0, 2):
        ref = _central_difference(lut, points, d)
        np.testing.assert_allclose(
            grads["q"][:, d].numpy(), ref, rtol=GRAD_RTOL, atol=GRAD_ATOL
        )


def test_gradient_dims_subset_selects_requested_axes():
    grids, data = _make_lut(shape=(5, 4, 3), n_wl=12)
    lut = BatchedLUT(grids, {"q": data})

    rng = np.random.default_rng(106)
    points = torch.as_tensor(_interior_points(grids, 24, rng))

    _, full = lut.interpolate_with_gradients(points)
    _, subset = lut.interpolate_with_gradients(points, dims=[2, 0])

    assert subset["q"].shape == (24, 2, 12)
    np.testing.assert_array_equal(subset["q"][:, 0].numpy(), full["q"][:, 2].numpy())
    np.testing.assert_array_equal(subset["q"][:, 1].numpy(), full["q"][:, 0].numpy())


def test_gradients_for_multiple_quantities_are_sliced_correctly():
    """Each quantity's gradient must come from its own columns of the stack."""
    grids, _ = _make_lut()
    rng = np.random.default_rng(107)
    shape = tuple(len(g) for g in grids)
    quantities = {f"q{i}": rng.normal(size=(*shape, 9)) for i in range(4)}

    fused = BatchedLUT(grids, quantities)
    points = torch.as_tensor(_interior_points(grids, 32, rng))
    _, grads = fused.interpolate_with_gradients(points)

    for key, arr in quantities.items():
        alone = BatchedLUT(grids, {key: arr})
        _, ref = alone.interpolate_with_gradients(points)
        np.testing.assert_allclose(
            grads[key].numpy(), ref[key].numpy(), rtol=FP64_RTOL, atol=FP64_RTOL
        )


def test_constants_are_absent_from_the_gradient_dict():
    """A constant quantity has an identically zero derivative and is not listed."""
    grids, data = _make_lut()
    lut = BatchedLUT(grids, {"q": data}, constants={"sphalb": 0.125})

    values, grads = lut.interpolate_with_gradients(
        torch.as_tensor(np.array([[0.5, 0.5, 0.5]]))
    )
    assert values["sphalb"] == 0.125
    assert "sphalb" not in grads
    assert set(grads) == {"q"}


def test_gradient_of_out_of_range_dim_raises():
    grids, data = _make_lut()
    lut = BatchedLUT(grids, {"q": data})
    with pytest.raises(ValueError, match="out of range"):
        lut.interpolate_with_gradients(
            torch.as_tensor(np.array([[0.5, 0.5, 0.5]])), dims=[3]
        )


def test_gradient_single_point_input_is_accepted():
    grids, data = _make_lut()
    lut = BatchedLUT(grids, {"q": data})
    values, grads = lut.interpolate_with_gradients(
        torch.as_tensor(np.array([0.3, 0.4, 0.5]))
    )
    assert values["q"].shape == (1, data.shape[-1])
    assert grads["q"].shape == (1, 3, data.shape[-1])


def test_float32_gradients_stay_within_tolerance():
    """fp32 tables: grid/delta math stays in coord_dtype, as for the values."""
    grids, data = _make_lut()
    lut64 = BatchedLUT(grids, {"q": data})
    lut32 = BatchedLUT(grids, {"q": data}, dtype=torch.float32)

    rng = np.random.default_rng(108)
    points = torch.as_tensor(_interior_points(grids, 64, rng))

    _, g64 = lut64.interpolate_with_gradients(points)
    _, g32 = lut32.interpolate_with_gradients(points)

    assert g32["q"].dtype == torch.float32
    np.testing.assert_allclose(
        g32["q"].numpy(), g64["q"].numpy(), rtol=FP32_RTOL, atol=1e-4
    )


# --- precision ------------------------------------------------------------------


def test_float32_stays_within_tolerance():
    """fp32 tables are allowed; grid math stays fp64 so cell choice is stable."""
    grids, data = _make_lut()
    rng = np.random.default_rng(4)
    points = np.stack([rng.uniform(g[0], g[-1], 200) for g in grids], axis=1)
    _assert_parity(grids, data, points, rtol=FP32_RTOL, dtype=torch.float32)


def test_float32_selects_same_cells_at_nodes():
    """The fp32 path must not flip cells near nodes, which would be discontinuous."""
    grids, data = _make_lut()
    node = np.array([g[1] for g in grids])
    points = np.stack([node - 1e-9, node, node + 1e-9])
    _assert_parity(grids, data, points, rtol=FP32_RTOL, dtype=torch.float32)


# --- input validation -----------------------------------------------------------


def test_wrong_point_dimensionality_raises():
    grids, data = _make_lut()
    lut = BatchedLUT(grids, {"q": data})
    with pytest.raises(ValueError, match="expected 3"):
        lut.interpolate(torch.zeros((4, 2), dtype=torch.float64))


def test_mismatched_grid_shape_raises():
    grids, data = _make_lut(shape=(5, 4, 3))
    with pytest.raises(ValueError, match="grid shape"):
        BatchedLUT(grids, {"q": data[:-1]})


def test_mismatched_channel_count_raises():
    grids, data = _make_lut()
    shape = tuple(len(g) for g in grids)
    with pytest.raises(ValueError, match="channels"):
        BatchedLUT(grids, {"a": data, "b": np.zeros((*shape, 3))})


def test_empty_data_raises():
    grids, _ = _make_lut()
    with pytest.raises(ValueError, match="at least one gridded quantity"):
        BatchedLUT(grids, {})


# --- housekeeping ---------------------------------------------------------------


def test_nbytes_reports_table_size():
    grids, data = _make_lut(shape=(5, 4, 3), n_wl=12)
    lut = BatchedLUT(grids, {"q": data})
    assert lut.nbytes == 5 * 4 * 3 * 12 * 8


def test_to_dtype_converts_table():
    grids, data = _make_lut()
    lut = BatchedLUT(grids, {"q": data})
    lut.to(dtype=torch.float32)
    out = lut.interpolate(torch.as_tensor(np.array([[0.5, 0.5, 0.5]])))["q"]
    assert out.dtype == torch.float32


# --- accelerator parity ---------------------------------------------------------
#
# The CPU tests above establish correctness of the algorithm. These re-run the
# same comparisons on a real accelerator, where a different kernel implementation
# of searchsorted/index_select is used and could diverge.


@pytest.mark.gpu
@requires_gpu
def test_accelerator_matches_reference():
    device = _accelerator()
    dtype, rtol = torch.float64, FP64_RTOL

    grids, data = _make_lut(shape=(6, 5, 4), n_wl=16)
    rng = np.random.default_rng(31)
    # Deliberately spans outside the grid so clamping is exercised on-device.
    points = np.stack([rng.uniform(-0.3, 1.3, 2000) for _ in grids], axis=1)

    ref = _reference(grids, data, points)
    got = (
        BatchedLUT(grids, {"q": data}, device=device, dtype=dtype)
        .interpolate(torch.as_tensor(points))["q"]
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=rtol)


@pytest.mark.gpu
@requires_gpu
def test_accelerator_gradients_match_central_difference():
    """The gradient's extra arithmetic, re-checked on device.

    Uses the same helper as the CPU test, so this is purely about the device
    kernels and not a second copy of the assertion.
    """
    device = _accelerator()
    grids, data = _make_lut(shape=(5, 4, 3), n_wl=16)
    lut = BatchedLUT(grids, {"q": data}, device=device, dtype=torch.float64)

    rng = np.random.default_rng(109)
    _assert_gradients_match_central_difference(
        lut, _interior_points(grids, 256, rng)
    )


@pytest.mark.gpu
@requires_gpu
def test_accelerator_handles_exact_nodes_and_singleton_dims():
    """On-device node lookups, against the analytic ground truth.

    Asserts against the table values rather than the numba reference: a node
    lookup must return that node's value, and ``VectorInterpolator.__call__``
    has been observed returning a blended value for this case on linux/x86.
    Comparing to the analytic answer keeps this test meaningful regardless.
    """
    device = _accelerator()
    dtype, rtol = torch.float64, FP64_RTOL

    grids, data = _make_lut(shape=(4, 1, 3))
    nodes, values = _node_points_and_values(grids, data)

    got = (
        BatchedLUT(grids, {"q": data}, device=device, dtype=dtype)
        .interpolate(torch.as_tensor(nodes))["q"]
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(got, values, rtol=rtol, atol=rtol)
