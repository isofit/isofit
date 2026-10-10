"""Both workers over the same block, on a real glint scene.

This is the layer nothing else executes. The unit tests cover the solve, the
surface and the initializer; ``test_torch_glint_example.py`` covers the solver
against a real ``ForwardModel``. None of them run ``TorchWorker.run_chunks``,
which is where the gather, the scatter, the output-block seeding and the
non-reflectance cube all meet -- and where the review demonstrated mutations
that corrupt output while every other test stays green.

The comparison is against ``analytical_line.Worker`` on identical inputs, with
the output cubes diffed on disk. Measured: reflectance 2.98e-08 (below one
float32 ULP of the output format), uncertainty 4.66e-10, and both glint cubes
bit-identical.

Verified to catch, among others, the two mutations the review showed survived
every other test: writing the non-reflectance cube as zeros, and not writing it
at all.

WHAT IT CANNOT SEE. Every pixel here belongs to one surface class, so there is
no other class's output in the block to preserve -- seeding the output blocks
from ``np.zeros`` instead of from disk passes this test. That invariant needs
two classes sharing a block and is covered by
``test_torch_multistate_blocks.py``.

Marked ``examples`` -- it needs the downloaded PRISM fixture and does not run
in CI. See ``test_torch_glint_example.py`` for setup.
"""

import copy
import os
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isofit.data import env  # noqa: E402

pytestmark = pytest.mark.examples

EXAMPLE = "20231110_Prism_Multisurface"
FLIGHT = "prm20231110t071521"
CONFIG = f"configs/{FLIGHT}_multi_surface_isofit.json"

# The output cubes are float32, so agreement is bounded by the write format.
RFL_ATOL = 1e-7
UNC_ATOL = 1e-8


def _root():
    try:
        path = env.path("examples", EXAMPLE)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"example path unavailable: {exc}")
    if not os.path.isfile(os.path.join(path, CONFIG)):
        pytest.skip(f"{EXAMPLE} not downloaded/built")
    return path


@pytest.fixture(scope="module")
def worker_outputs(tmp_path_factory):
    """Run both workers over the whole scene and return the four cube pairs."""
    from spectral.io import envi

    from isofit.configs import configs
    from isofit.core.common import envi_header
    from isofit.core.fileio import write_bil_chunk
    from isofit.core.forward import ForwardModel
    from isofit.core.multistate import construct_full_state, update_config_for_surface

    root = _root()
    ex = os.path.join(root, "remote", FLIGHT)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = configs.create_new_config(os.path.join(root, CONFIG))
        base.forward_model.instrument.integrations = 1
        full_sv, f_surf, f_rfl, _, f_atm, _ = construct_full_state(base)
        sub_cfg = update_config_for_surface(
            copy.deepcopy(base), "glint_model_surface"
        )
        fm = ForwardModel(sub_cfg)

    n_rfl = len(f_rfl)
    n_non_rfl = len(f_surf) - n_rfl
    assert n_non_rfl == 2, f"expected SKY_GLINT and SUN_GLINT, got {n_non_rfl}"

    rdn = envi.open(envi_header(ex + "_rdn_two_px")).open_memmap(interleave="bip")
    lines, samples, _ = rdn.shape
    tmp = str(tmp_path_factory.mktemp("worker_e2e"))

    def make(name, bands, data=None, nl=lines, ns=samples):
        path = os.path.join(tmp, name)
        envi.create_image(
            envi_header(path),
            {"lines": nl, "samples": ns, "bands": bands, "interleave": "bil",
             "data type": 4, "byte order": 0, "header offset": 0},
            ext="", force=True,
        )
        cube = np.full((nl, ns, bands), -9999.0) if data is None else data
        write_bil_chunk(np.swapaxes(cube, 1, 2), path, 0, (nl, bands, ns))
        return path

    atm = np.broadcast_to(
        fm.init[fm.idx_atmosphere], (lines, samples, len(f_atm))
    ).copy()
    atm_file = make("atm_interp", len(f_atm), data=atm)

    init = fm.init
    if len(init) < len(full_sv):
        init = np.pad(init, (0, len(full_sv) - len(init)))
    subs = np.broadcast_to(init[: len(full_sv)], (1, 1, len(full_sv))).copy()
    subs_file = make("subs_state", len(full_sv), data=subs, nl=1, ns=1)
    lbl_file = make("lbl", 1, data=np.zeros((lines, samples, 1)))

    outs = {
        tag: {
            "rfl": make(f"{tag}_rfl", n_rfl),
            "unc": make(f"{tag}_unc", n_rfl),
            "nrfl": make(f"{tag}_nrfl", n_non_rfl),
            "nrfl_unc": make(f"{tag}_nrfl_unc", n_non_rfl),
        }
        for tag in ("numpy", "torch")
    }

    class_idx = np.array([[r, c] for r in range(lines) for c in range(samples)])

    def wargs(o):
        return (
            sub_cfg, fm, "glint_model_surface", class_idx, full_sv,
            f_surf, f_rfl, f_atm,
            ex + "_rdn_two_px", ex + "_loc_two_px", ex + "_obs_two_px",
            atm_file, subs_file, lbl_file,
            o["rfl"], o["unc"], o["nrfl"], o["nrfl_unc"],
            1, "ERROR", None, "algebraic", None, None,
        )

    from isofit.utils.analytical_line import Worker
    from isofit.utils.analytical_line_torch import TorchWorker

    def unwrap(cls):
        # Under the ray bypass the class is wrapped; reach the real one.
        return getattr(cls, "__ray_actor_class__", None) or getattr(cls, "obj", cls)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unwrap(Worker)(*wargs(outs["numpy"])).run_chunks((0, lines))
        unwrap(TorchWorker)(
            *wargs(outs["torch"]), "cpu", "auto", "float64"
        ).run_chunks((0, lines))

    def read(path):
        return np.array(
            envi.open(envi_header(path)).open_memmap(
                interleave="bip", writable=False
            )
        )

    return {k: {t: read(outs[t][k]) for t in ("numpy", "torch")}
            for k in ("rfl", "unc", "nrfl", "nrfl_unc")}


def _diff(pair):
    a, b = pair["numpy"], pair["torch"]
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    m = np.isfinite(a) & np.isfinite(b)
    assert m.any(), "no comparable values"
    return np.abs(a[m] - b[m]).max()


def test_reflectance_cube_matches(worker_outputs):
    assert _diff(worker_outputs["rfl"]) < RFL_ATOL


def test_uncertainty_cube_matches(worker_outputs):
    assert _diff(worker_outputs["unc"]) < UNC_ATOL


def test_glint_cube_matches_bit_for_bit(worker_outputs):
    """The non-reflectance cube is the one the torch worker used not to write."""
    pair = worker_outputs["nrfl"]
    assert _diff(pair) == 0.0, "SKY/SUN_GLINT diverged"


def test_glint_uncertainty_cube_matches_bit_for_bit(worker_outputs):
    assert _diff(worker_outputs["nrfl_unc"]) == 0.0


def test_glint_cube_is_actually_populated(worker_outputs):
    """Guard against agreement by both cubes being left at the fill value."""
    torch_cube = worker_outputs["nrfl"]["torch"]
    assert (torch_cube != -9999.0).any(), "the glint cube is all fill"
    assert np.ptp(torch_cube) > 1.0, (
        f"glint cube spans only {np.ptp(torch_cube):.3g}; it looks constant"
    )
