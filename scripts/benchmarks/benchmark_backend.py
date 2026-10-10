#! /usr/bin/env python3
"""
Benchmark the ISOFIT `analytical_line` stage: numpy backend vs torch backend.

Both backends run the same scene, from the same config, on the same host, in the
same session. That is deliberate -- BLAS build, clock speed, thermal state, and
disk cache all move the absolute numbers around, and running the two legs
back-to-back is what makes their ratio meaningful. Numbers copied between hosts
are not comparable.

Every stage invocation is a subprocess, so each run gets a clean ray cluster, a
clean CUDA context, and a real exit code. Runs are validated before anything is
reported: a benchmark that silently times a crashed or empty run is worse than
no benchmark at all, because it looks like data.

What is timed
-------------
Wall-clock time of the whole `analytical_line` invocation (process start to
exit), which includes ray startup, config parsing, and output initialization.
The stage's own inversion timer is parsed out of the logs and reported next to
it as a secondary metric. Atmospheric interpolation is *excluded*: it is a
numpy-only preprocessing step, identical for both backends, so the harness
points every run at one shared `--atm_file` that the warmup run creates.

Usage
-----
    python scripts/benchmarks/benchmark_backend.py \
        RDN_FILE LOC_FILE OBS_FILE ISOFIT_DIR \
        --isofit_config CONFIG.json \
        --backends numpy,torch \
        --torch_device cuda \
        --torch_batch_size 4096 \
        --repeats 3 \
        --output_dir bench_out \
        --output_json benchmark_result.json

The JSON artifact is versioned (see SCHEMA_VERSION) and carries an environment
block, so a result file is interpretable without the shell history that made it.
"""

import json
import multiprocessing
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
from spectral.io import envi

import isofit
from isofit.core.common import envi_header

# Bump the minor version for additive fields, the major for anything that moves
# or removes one. Downstream readers should check this before trusting keys.
SCHEMA_VERSION = "isofit-benchmark-backend/1.0"

# analytical_line writes this into pixels it skipped (all-negative radiance).
FILL_VALUE = -9999.0

# analytical_line logs: "Analytical line inversions complete.  12.3s total,
# 567.8 spectra/s, ..."
STAGE_RE = re.compile(
    r"Analytical line inversions complete\.\s+([0-9.]+)s total,\s+([0-9.]+) spectra/s"
)
# analytical_line_torch: "TorchWorker on cuda:0: batch=4096, 5.0 MiB/pixel"
WORKER_RE = re.compile(r"TorchWorker on (.+?): batch=(\d+)")


class BenchmarkError(RuntimeError):
    """A run could not be trusted. Nothing gets reported."""


def abort(message: str):
    """Fail loudly. Never fall through to reporting a number."""
    bar = "*" * 88
    click.echo(f"\n{bar}\n! BENCHMARK ABORTED\n! {message}\n{bar}", err=True)
    raise SystemExit(2)


# --- environment ----------------------------------------------------------------


def git_describe() -> dict:
    """Repo SHA and dirty flag, or nulls outside a checkout."""
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        out = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True
        )
        return out.stdout.strip() if out.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "git_sha": sha,
        # A dirty tree means the artifact does not correspond to any commit.
        # Record it rather than pretending the SHA is the whole story.
        "git_dirty": None if status is None else bool(status),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def torch_environment(torch_device: str, needed: bool) -> dict:
    """Resolve device and dtype exactly the way the workers will.

    Doubles as a pre-flight: an explicitly requested device that is unavailable
    raises here, before an hour of scene processing, not after.
    """
    info = {
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": None,
        "cuda_device_count": None,
        "device_requested": torch_device,
        "device_resolved": None,
        "device_name": None,
        "dtype": None,
    }
    if not needed:
        return info

    try:
        import torch
    except ImportError:
        abort("the torch backend was requested but torch is not installed")

    from isofit.core.backend import resolve_device, resolve_dtype

    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_device_count"] = (
        torch.cuda.device_count() if info["cuda_available"] else 0
    )

    try:
        device = resolve_device(torch_device)
    except (RuntimeError, ValueError) as e:
        abort(f"device {torch_device!r} could not be resolved: {e}")

    info["device_resolved"] = str(device)
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(device.index or 0)
    else:
        info["device_name"] = platform.processor() or device.type
    info["dtype"] = str(resolve_dtype("auto", device)).replace("torch.", "")

    if device.type != "cuda":
        click.echo(
            f"WARNING: benchmarking torch on {device.type}. CUDA is the "
            "production target; cpu and mps numbers are plumbing checks only.",
            err=True,
        )
    return info


def environment_block(
    torch_device: str, torch_batch_size: str, n_cores: int, needs_torch: bool
) -> dict:
    debug = os.environ.get("ISOFIT_DEBUG")
    if debug == "1":
        # ray_bypass forces the whole run serial, which changes what the wall
        # clock means. Record it, and say so out loud.
        click.echo(
            "WARNING: ISOFIT_DEBUG=1 is set -- ray is bypassed and every run is "
            "serial. Timings are not representative of a parallel deployment.",
            err=True,
        )
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "isofit_version": isofit.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "n_cores": n_cores,
        "torch_batch_size_requested": torch_batch_size,
        "isofit_debug": debug,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    env.update(git_describe())
    env.update(torch_environment(torch_device, needs_torch))
    return env


# --- validation -----------------------------------------------------------------


def cube_stats(path: str, chunk: int = 64) -> dict:
    """Summarize a reflectance cube in line-blocks (cubes do not fit in RAM).

    Returns counts rather than a verdict; `validate_run` applies the policy.
    """
    header = envi_header(path)
    if not os.path.isfile(header) or not os.path.isfile(path):
        raise BenchmarkError(f"no output cube at {path} (header or data missing)")
    if os.path.getsize(path) == 0:
        raise BenchmarkError(f"output cube {path} is zero bytes")

    # BIP view: (lines, samples, bands).
    memmap = envi.open(header).open_memmap(interleave="bip")
    lines, samples, bands = memmap.shape

    total = finite = valid = plausible = 0
    vmin, vmax, vsum = np.inf, -np.inf, 0.0
    for start in range(0, lines, chunk):
        block = np.asarray(memmap[start : start + chunk, :, :], dtype=np.float64)
        total += block.size
        is_finite = np.isfinite(block)
        finite += int(is_finite.sum())
        # -9999 is written as float32 and is exactly representable, so equality
        # is safe here.
        is_valid = is_finite & (block != FILL_VALUE)
        valid += int(is_valid.sum())
        if is_valid.any():
            good = block[is_valid]
            # Open interval: reflectance of exactly 0 is what initialize_output
            # pre-fills, so 0 must not count as a retrieval (see validate_run).
            plausible += int(((good > 0.0) & (good < 1.0)).sum())
            vmin = min(vmin, float(good.min()))
            vmax = max(vmax, float(good.max()))
            vsum += float(good.sum())

    del memmap
    return {
        "path": str(path),
        "shape": [lines, samples, bands],
        "pixels": lines * samples,
        "n_elements": total,
        "frac_finite": finite / total if total else 0.0,
        "frac_valid": valid / total if total else 0.0,
        "frac_plausible_of_valid": plausible / valid if valid else 0.0,
        "min": None if vmin == np.inf else vmin,
        "max": None if vmax == -np.inf else vmax,
        "mean": (vsum / valid) if valid else None,
    }


def validate_run(
    record: dict,
    expected_pixels: int,
    min_valid_fraction: float,
    min_plausible_fraction: float,
) -> dict:
    """Refuse to time a run that did not actually retrieve anything.

    Three distinct silent-failure modes are covered, in order of sneakiness:

    1. Nonzero exit / timeout. Cheapest to detect, checked first.
    2. Missing, truncated, or wrong-shaped cube. A worker that died after the
       driver initialized the outputs leaves a plausible-looking file behind.
    3. A well-formed cube full of zeros. `initialize_output` pre-fills every
       output with zeros, so a run whose workers all raised still produces a
       complete, correctly-shaped, correctly-headed ENVI cube of zeros -- and
       it finishes *fast*, which is exactly the direction that flatters a
       benchmark. The open-interval (0, 1) test is what catches this: zero is
       not a physically plausible retrieved reflectance, and neither is a fill
       value, a NaN, nor a negative number.
    """
    tag = record["tag"]

    if record.get("timed_out"):
        raise BenchmarkError(
            f"run {tag} exceeded its timeout of {record['timeout']}s; see "
            f"{record['log']}"
        )
    if record["returncode"] != 0:
        raise BenchmarkError(
            f"run {tag} exited {record['returncode']}; see {record['log']}"
        )

    stats = cube_stats(record["rfl_file"])

    if stats["pixels"] != expected_pixels:
        raise BenchmarkError(
            f"run {tag} wrote {stats['pixels']} pixels, scene has "
            f"{expected_pixels}; the cube does not cover the scene"
        )
    if stats["frac_valid"] < min_valid_fraction:
        raise BenchmarkError(
            f"run {tag}: only {stats['frac_valid']:.1%} of the cube is valid "
            f"(finite and not fill), below the {min_valid_fraction:.0%} floor. "
            "The run completed but retrieved almost nothing."
        )
    if stats["frac_plausible_of_valid"] < min_plausible_fraction:
        raise BenchmarkError(
            f"run {tag}: only {stats['frac_plausible_of_valid']:.1%} of valid "
            f"values are in (0, 1), below the {min_plausible_fraction:.0%} "
            f"floor (min={stats['min']}, max={stats['max']}, "
            f"mean={stats['mean']}). This is not a physically plausible "
            "reflectance cube -- an all-zero cube from crashed workers looks "
            "exactly like this."
        )

    record["validation"] = stats
    return stats


def compare_cubes(reference: str, candidate: str, chunk: int = 64) -> dict:
    """Streaming difference between two reflectance cubes.

    Parity is gated by the pytest suite, not here; this is a coarse guard
    against benchmarking a backend that is fast because it is computing
    something else.
    """
    ref = envi.open(envi_header(reference)).open_memmap(interleave="bip")
    cand = envi.open(envi_header(candidate)).open_memmap(interleave="bip")
    if ref.shape != cand.shape:
        raise BenchmarkError(
            f"cube shapes differ: {reference} {ref.shape} vs {candidate} {cand.shape}"
        )

    lines = ref.shape[0]
    max_abs = 0.0
    diff_sum = 0.0
    count = 0
    for start in range(0, lines, chunk):
        a = np.asarray(ref[start : start + chunk], dtype=np.float64)
        b = np.asarray(cand[start : start + chunk], dtype=np.float64)
        both = np.isfinite(a) & np.isfinite(b) & (a != FILL_VALUE) & (b != FILL_VALUE)
        if not both.any():
            continue
        d = np.abs(a[both] - b[both])
        max_abs = max(max_abs, float(d.max()))
        diff_sum += float(d.sum())
        count += int(d.size)

    del ref, cand
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "compared_elements": count,
        "max_abs_diff": max_abs,
        "mean_abs_diff": (diff_sum / count) if count else None,
    }


# --- running the stage ----------------------------------------------------------


def scene_shape(rdn_file: str) -> dict:
    ds = envi.open(envi_header(rdn_file))
    lines, samples, bands = ds.shape
    del ds
    return {
        "rdn_file": str(rdn_file),
        "lines": lines,
        "samples": samples,
        "bands": bands,
        "pixels": lines * samples,
    }


def run_stage(tag: str, backend: str, opts: dict) -> dict:
    """Run one `isofit analytical_line` invocation and time it end to end.

    A subprocess (rather than an in-process call) so every run gets a fresh ray
    cluster and CUDA context -- and so a crash is an exit code instead of a
    half-initialized interpreter that poisons the next repeat.
    """
    output_dir = Path(opts["output_dir"])
    rfl_file = output_dir / f"{tag}_rfl"
    unc_file = output_dir / f"{tag}_uncert"
    log_file = output_dir / f"{tag}.log"

    cmd = [
        sys.executable,
        "-m",
        "isofit",
        "analytical_line",
        opts["rdn_file"],
        opts["loc_file"],
        opts["obs_file"],
        opts["isofit_dir"],
        "--output_rfl_file",
        str(rfl_file),
        "--output_unc_file",
        str(unc_file),
        "--atm_file",
        str(opts["atm_file"]),
        "--n_cores",
        str(opts["n_cores"]),
        "--backend",
        backend,
        "--loglevel",
        "INFO",
    ]
    if opts["isofit_config"]:
        cmd += ["--isofit_config", opts["isofit_config"]]
    if opts["segmentation_file"]:
        cmd += ["--segmentation_file", opts["segmentation_file"]]
    if backend == "torch":
        cmd += [
            "--torch_device",
            opts["torch_device"],
            "--torch_batch_size",
            str(opts["torch_batch_size"]),
        ]
    cmd += list(opts["extra"])

    click.echo(f"[{tag}] $ {' '.join(cmd)}", err=True)

    timed_out = False
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=opts["timeout"] or None,
        )
        elapsed = time.perf_counter() - start
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        timed_out = True
        returncode = -1
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")

    log_file.write_text(
        f"$ {' '.join(cmd)}\n\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
    )

    # ISOFIT logs to stderr; ray forwards actor logs to the driver, so the
    # worker lines usually land here too. Parse defensively -- these are
    # secondary metrics and their absence is not a failure.
    combined = f"{stdout}\n{stderr}"
    stage = STAGE_RE.search(combined)
    worker = WORKER_RE.search(combined)

    return {
        "tag": tag,
        "backend": backend,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout": opts["timeout"],
        "seconds": elapsed,
        "stage_seconds": float(stage.group(1)) if stage else None,
        "stage_spectra_per_s": float(stage.group(2)) if stage else None,
        "resolved_device": worker.group(1) if worker else None,
        "resolved_batch_size": int(worker.group(2)) if worker else None,
        "rfl_file": str(rfl_file),
        "unc_file": str(unc_file),
        "log": str(log_file),
        "command": cmd,
    }


def summarize(backend: str, runs: list, pixels: int) -> dict:
    """Median over the timed repeats.

    Median, not mean: one slow repeat (a checkpoint, a noisy neighbor, a page
    cache miss) should not move the headline number.
    """
    seconds = [r["seconds"] for r in runs]
    median_seconds = statistics.median(seconds)
    stage_seconds = [r["stage_seconds"] for r in runs if r["stage_seconds"]]
    last = runs[-1]
    return {
        "backend": backend,
        "repeats": len(runs),
        "pixels": pixels,
        "seconds": seconds,
        "median_seconds": median_seconds,
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        # Primary metric.
        "spectra_per_s": pixels / median_seconds if median_seconds else None,
        # Secondary: the stage's own inversion timer, excluding process start,
        # ray init, and output allocation.
        "median_stage_seconds": statistics.median(stage_seconds)
        if stage_seconds
        else None,
        "median_stage_spectra_per_s": (pixels / statistics.median(stage_seconds))
        if stage_seconds
        else None,
        "resolved_device": last["resolved_device"],
        "resolved_batch_size": last["resolved_batch_size"],
        "reference_cube": last["rfl_file"],
    }


# --- cli ------------------------------------------------------------------------


@click.command(name="benchmark_backend")
@click.argument("rdn_file", type=click.Path(exists=True))
@click.argument("loc_file", type=click.Path(exists=True))
@click.argument("obs_file", type=click.Path(exists=True))
@click.argument("isofit_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--isofit_config", type=click.Path(exists=True), default=None)
@click.option("--segmentation_file", type=click.Path(exists=True), default=None)
@click.option(
    "--backends",
    default="numpy,torch",
    help="Comma-separated backends to benchmark, in order. The first is the reference.",
)
@click.option("--repeats", type=int, default=3, help="Timed runs per backend.")
@click.option(
    "--warmup/--no_warmup",
    default=True,
    help="Run one untimed warmup per backend (numba JIT, CUDA context, LUT upload, "
    "atmospheric interpolation, page cache). Disabling it biases the first repeat.",
)
@click.option("--n_cores", type=int, default=-1, help="-1 uses every core.")
@click.option("--torch_device", default="auto", help="auto, cpu, mps, cuda, or cuda:N.")
@click.option("--torch_batch_size", default="auto", help="Spectra per batched call.")
@click.option(
    "--atm_file",
    type=click.Path(),
    default=None,
    help="Shared atmospheric interpolation file. Defaults to one inside "
    "--output_dir, created by the warmup run so timed runs measure retrieval only.",
)
@click.option("--output_dir", type=click.Path(), default="benchmark_output")
@click.option("--output_json", type=click.Path(), default="benchmark_result.json")
@click.option("--timeout", type=int, default=0, help="Per-run seconds; 0 disables.")
@click.option(
    "--min_valid_fraction",
    type=float,
    default=0.5,
    help="Minimum fraction of cube elements that must be finite and non-fill.",
)
@click.option(
    "--min_plausible_fraction",
    type=float,
    default=0.9,
    help="Minimum fraction of valid values that must fall in the open interval (0, 1).",
)
@click.option(
    "--parity_atol",
    type=float,
    default=1e-4,
    help="Max absolute reflectance difference from the reference backend.",
)
@click.option(
    "--strict_parity",
    is_flag=True,
    help="Abort instead of warning when --parity_atol is exceeded.",
)
@click.option(
    "--extra",
    multiple=True,
    help="Extra argument forwarded verbatim to analytical_line; repeatable.",
)
def cli(
    rdn_file,
    loc_file,
    obs_file,
    isofit_dir,
    isofit_config,
    segmentation_file,
    backends,
    repeats,
    warmup,
    n_cores,
    torch_device,
    torch_batch_size,
    atm_file,
    output_dir,
    output_json,
    timeout,
    min_valid_fraction,
    min_plausible_fraction,
    parity_atol,
    strict_parity,
    extra,
):
    """Benchmark analytical_line across numerical backends on one fixed scene."""
    backend_list = [b.strip() for b in backends.split(",") if b.strip()]
    unknown = [b for b in backend_list if b not in ("numpy", "torch")]
    if unknown:
        abort(f"unknown backend(s): {unknown}. Valid: numpy, torch.")
    if repeats < 1:
        abort(f"--repeats must be at least 1, got {repeats}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if atm_file is None:
        atm_file = output_dir / "benchmark_atm_interp"

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()

    if not warmup and not os.path.isfile(atm_file):
        click.echo(
            "WARNING: --no_warmup with no existing --atm_file. The first timed "
            "run of the first backend will also pay for atmospheric "
            "interpolation and is not comparable to the rest.",
            err=True,
        )

    # Resolve the device first: an unavailable explicit device should fail in
    # the first second, not after the scene has been opened and a run started.
    environment = environment_block(
        torch_device, torch_batch_size, n_cores, needs_torch="torch" in backend_list
    )

    try:
        scene = scene_shape(rdn_file)
    except Exception as e:
        abort(f"could not read the radiance cube {rdn_file}: {e}")
    pixels = scene["pixels"]

    opts = {
        "rdn_file": rdn_file,
        "loc_file": loc_file,
        "obs_file": obs_file,
        "isofit_dir": isofit_dir,
        "isofit_config": isofit_config,
        "segmentation_file": segmentation_file,
        "atm_file": atm_file,
        "n_cores": n_cores,
        "torch_device": torch_device,
        "torch_batch_size": torch_batch_size,
        "output_dir": output_dir,
        "timeout": timeout,
        "extra": extra,
    }

    click.echo(
        f"Scene: {scene['lines']}x{scene['samples']} = {pixels} pixels, "
        f"{scene['bands']} bands",
        err=True,
    )

    all_runs = []
    results = []
    reference_cube = None

    try:
        for backend in backend_list:
            timed = []

            if warmup:
                # Excluded from every statistic: the first run pays numba JIT,
                # the CUDA context and LUT upload, atmospheric interpolation,
                # and a cold page cache. Validated anyway -- a warmup that
                # crashed means the timed runs are about to as well.
                record = run_stage(f"{backend}_warmup", backend, opts)
                record["warmup"] = True
                all_runs.append(record)
                validate_run(
                    record, pixels, min_valid_fraction, min_plausible_fraction
                )
                click.echo(
                    f"[{backend}] warmup {record['seconds']:.1f}s (excluded)", err=True
                )

            for i in range(repeats):
                record = run_stage(f"{backend}_{i}", backend, opts)
                record["warmup"] = False
                all_runs.append(record)
                validate_run(
                    record, pixels, min_valid_fraction, min_plausible_fraction
                )
                timed.append(record)
                click.echo(
                    f"[{backend}] repeat {i}: {record['seconds']:.1f}s, "
                    f"{pixels / record['seconds']:.1f} spectra/s",
                    err=True,
                )

            summary = summarize(backend, timed, pixels)

            # Every backend must agree with the reference. A backend that is
            # fast because it solved a different problem is not a speedup.
            if reference_cube is None:
                reference_cube = summary["reference_cube"]
                summary["parity"] = None
            else:
                parity = compare_cubes(reference_cube, summary["reference_cube"])
                summary["parity"] = parity
                if parity["max_abs_diff"] > parity_atol:
                    message = (
                        f"{backend} differs from the reference backend by "
                        f"{parity['max_abs_diff']:.3e} reflectance (tolerance "
                        f"{parity_atol:.1e})"
                    )
                    if strict_parity:
                        raise BenchmarkError(message)
                    click.echo(f"WARNING: {message}", err=True)

            results.append(summary)

    except BenchmarkError as e:
        abort(str(e))

    comparison = None
    if len(results) > 1:
        baseline, candidate = results[0], results[-1]
        comparison = {
            "baseline": baseline["backend"],
            "candidate": candidate["backend"],
            "wall_clock_speedup": baseline["median_seconds"]
            / candidate["median_seconds"],
            "stage_speedup": (
                baseline["median_stage_seconds"] / candidate["median_stage_seconds"]
            )
            if baseline["median_stage_seconds"] and candidate["median_stage_seconds"]
            else None,
        }

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "scene": scene,
        "invocation": {
            "backends": backend_list,
            "repeats": repeats,
            "warmup": warmup,
            "isofit_dir": str(isofit_dir),
            "isofit_config": str(isofit_config) if isofit_config else None,
            "atm_file": str(atm_file),
            "output_dir": str(output_dir),
            "min_valid_fraction": min_valid_fraction,
            "min_plausible_fraction": min_plausible_fraction,
            "parity_atol": parity_atol,
            "extra": list(extra),
        },
        "results": results,
        "comparison": comparison,
        "runs": all_runs,
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(artifact, indent=2) + "\n")

    click.echo("")
    click.echo(
        f"{'backend':>8}  {'device':>12}  {'pixels':>10}  "
        f"{'median s':>10}  {'spectra/s':>12}"
    )
    for r in results:
        # numpy never logs a device; it is the ray CPU pool by construction.
        device = r["resolved_device"] or ("cpu" if r["backend"] == "numpy" else "-")
        click.echo(
            f"{r['backend']:>8}  {device:>12}  "
            f"{r['pixels']:>10}  {r['median_seconds']:>10.1f}  "
            f"{r['spectra_per_s']:>12.1f}"
        )
    if comparison:
        click.echo(
            f"\n{comparison['candidate']} vs {comparison['baseline']}: "
            f"{comparison['wall_clock_speedup']:.2f}x wall clock"
            + (
                f", {comparison['stage_speedup']:.2f}x inversion-only"
                if comparison["stage_speedup"]
                else ""
            )
        )
    click.echo(f"\nWrote {output_json}")


if __name__ == "__main__":
    cli()
