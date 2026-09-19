"""The backend options must actually reach the worker that uses them.

``torch_dtype`` was validated by the config section, returned by
``resolve_backend_options``, documented as a user-facing option -- and never
passed to ``TorchWorker``, which hardcoded ``resolve_dtype("auto", ...)``.
Selecting float32 on the analytical-line path silently did nothing.

Nothing in the unit or parity suites could catch that: every test either
constructs the solver directly (bypassing the worker) or runs on CPU, where
"auto" and "float64" resolve to the same dtype. The gap is in the *wiring*, so
the test has to be about the wiring.

The dispatch site passes arguments positionally (``*wargs`` followed by the
backend options), which is also an ordering hazard: inserting a parameter in
``TorchWorker.__init__`` would shift the batch size into the dtype slot with no
error, just a wrong run. These tests bind the actual call-site arguments to the
actual constructor signature and assert where each one lands.
"""

import ast
import inspect
from pathlib import Path

import pytest

from isofit.core.backend import resolve_backend_options
from isofit.utils import analytical_line
from isofit.utils.analytical_line_torch import TorchWorker

pytestmark = pytest.mark.torch_cpu

BACKEND_OPTIONS = ("torch_device", "torch_batch_size", "torch_dtype")


def _worker_signature():
    """The undecorated ``TorchWorker.__init__`` signature.

    ``@ray.remote`` wraps the class, so reach through to the original when the
    real ray is installed and fall back to the class itself under the bypass.
    """
    cls = getattr(TorchWorker, "__ray_actor_class__", TorchWorker)
    params = list(inspect.signature(cls.__init__).parameters)
    return params[1:] if params and params[0] == "self" else params


def _dispatch_call():
    """The ``TorchWorker...remote(...)`` call node in analytical_line.py."""
    tree = ast.parse(Path(inspect.getfile(analytical_line)).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
        and "TorchWorker" in ast.unparse(node.func)
    ]
    assert len(calls) == 1, f"expected one TorchWorker.remote() call, found {len(calls)}"
    return calls[0]


def _static_length(node):
    """Number of elements a list expression contributes, or None if unknowable.

    Covers the two forms the dispatch actually uses: a list literal, and a
    comprehension over a literal tuple (``[ray.put(o) for o in (config, fm)]``).
    """
    if isinstance(node, ast.List):
        return len(node.elts)
    if isinstance(node, ast.ListComp) and len(node.generators) == 1:
        source = node.generators[0]
        if not source.ifs and isinstance(source.iter, (ast.Tuple, ast.List)):
            return len(source.iter.elts)
    return None


def _wargs_length():
    """Number of entries in the shared ``wargs`` list the dispatch splats."""
    tree = ast.parse(Path(inspect.getfile(analytical_line)).read_text())
    total = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "wargs" for t in node.targets
        ):
            total = _static_length(node.value)
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "wargs"
            and isinstance(node.op, ast.Add)
            and total is not None
        ):
            extra = _static_length(node.value)
            total = None if extra is None else total + extra
    if total is None:
        pytest.fail("could not statically size the wargs list in analytical_line.py")
    return total


def test_worker_accepts_every_backend_option():
    """Each option resolve_backend_options returns must be a worker parameter."""
    params = _worker_signature()
    for option in BACKEND_OPTIONS:
        assert option in params, f"TorchWorker cannot receive {option}"


def test_dispatch_passes_every_backend_option():
    """Each worker-configuring option must appear at the dispatch site.

    This is the regression test for the original defect: ``torch_dtype`` was a
    documented, validated option that the dispatch simply never handed over, so
    the worker fell back to a hardcoded default. ``torch_num_gpu_workers`` is
    deliberately excluded -- it sizes the actor pool rather than a worker.
    """
    passed = {
        ast.literal_eval(arg.slice)
        for arg in _dispatch_call().args
        if not isinstance(arg, ast.Starred)
    }
    missing = set(BACKEND_OPTIONS) - passed
    assert not missing, f"dispatch never passes {sorted(missing)} to TorchWorker"


def test_dispatch_passes_options_into_the_matching_parameters():
    """The positional arguments must land on the parameters they are named for.

    This is the ordering check: it binds ``len(wargs)`` placeholders plus the
    literal trailing arguments from the call site to the real signature, and
    asserts each ``opts[...]`` subscript arrives at the identically-named
    parameter.
    """
    call = _dispatch_call()
    params = _worker_signature()

    assert not call.keywords, (
        "dispatch now passes keywords; this test only models positional args"
    )
    starred = [a for a in call.args if isinstance(a, ast.Starred)]
    assert len(starred) == 1 and ast.unparse(starred[0].value) == "wargs"

    # Positional layout: wargs expands to N arguments, then the trailing ones.
    trailing = [a for a in call.args if not isinstance(a, ast.Starred)]
    offset = _wargs_length()

    for i, arg in enumerate(trailing):
        source = ast.unparse(arg)
        assert source.startswith("opts["), (
            f"unexpected trailing dispatch argument {source!r}"
        )
        option = ast.literal_eval(arg.slice)
        landed_on = params[offset + i]
        assert landed_on == option, (
            f"opts[{option!r}] is passed in position {offset + i}, which is the "
            f"{landed_on!r} parameter"
        )


def test_every_passed_option_is_one_resolve_backend_options_produces():
    """No dispatch argument may name an option the resolver never returns."""
    produced = set(resolve_backend_options(None))
    for arg in _dispatch_call().args:
        if isinstance(arg, ast.Starred):
            continue
        assert ast.literal_eval(arg.slice) in produced


def test_worker_resolves_the_dtype_it_was_given():
    """The dtype parameter must feed resolve_dtype, not a hardcoded default."""
    cls = getattr(TorchWorker, "__ray_actor_class__", TorchWorker)
    source = inspect.getsource(cls.__init__)
    assert "resolve_dtype(torch_dtype" in source, (
        "TorchWorker ignores its torch_dtype argument"
    )
