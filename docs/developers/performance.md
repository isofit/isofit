# Performance

ISOFIT's default numerical backend is numpy, and it inverts one pixel at a time. There is also an opt-in `torch` backend that batches the analytical-line retrieval so that many pixels are solved per call, which is what makes a GPU worth using. The numpy path is untouched by any of this: the torch backend is selected only when it is explicitly requested.

Backend | Description
-|-
`numpy` | Default. Scalar, per-pixel retrieval. Runs everywhere, no extra dependencies.
`torch` | Opt-in. Batches pixels through the analytical solve on a device. CUDA is the production target.

???+ note

    Only the analytical-line retrieval has a torch path (`isofit analytical_line`, and the analytical-line stage of `isofit apply_oe`). Every other stage runs on numpy regardless of this setting.

## Device selection

Devices are resolved by `isofit.core.backend.resolve_device`, which is the single source of truth for the policy below.

`torch_device` | Behavior
-|-
`auto` | Prefer `cuda`, then `mps`, then `cpu`.
`cpu` | Always available. Selected silently, since asking for it is an explicit choice.
`mps` | Raises if Metal Performance Shaders are unavailable. Warns about precision (see below).
`cuda` / `cuda:N` | Raises if CUDA is unavailable, or if index `N` exceeds the visible device count.

Two rules govern the behavior, and both exist to keep a performance request from turning into a silent slow success:

* **An explicit device is never downgraded.** Requesting `cuda` on a host with a CPU-only torch build raises a `RuntimeError` that names the likely cause. A failure is much easier to notice than a run that quietly took twenty times longer than expected.
* **`auto` falling back to CPU is loud.** When no accelerator is found, `auto` resolves to `cpu` and prints a banner warning, because torch on CPU is expected to be *slower* than the numpy backend for retrievals. Set the device explicitly to `cpu` to silence it.

### Precision

The retrieval math needs float64: it Cholesky-factorizes measurement covariances and takes finite-difference Jacobians with `eps=1e-5`. `torch_dtype: auto` therefore resolves to float64 on `cuda` and `cpu`.

MPS does not implement float64 at all, so `mps` resolves to float32, and asking for `float64` on `mps` is a configuration error. **MPS is not a supported target.** It is useful for plumbing work on Apple silicon; no parity or performance number should be quoted from it.

## Configuration

The options live in the `implementation` section of the ISOFIT config.

Option | Default | Description
-|-|-
`backend` | `"numpy"` | Numerical backend: `"numpy"` or `"torch"`.
`torch_device` | `"auto"` | `"auto"`, `"cpu"`, `"mps"`, `"cuda"`, or `"cuda:N"`. Ignored when the backend is numpy.
`torch_batch_size` | `"auto"` | Spectra per batched call, as a positive integer or `"auto"`. `"auto"` sizes the batch against free device memory.
`torch_dtype` | `"auto"` | `"auto"`, `"float32"`, or `"float64"`. `"auto"` is float64 everywhere except mps.
`torch_num_gpu_workers` | `null` | Number of GPU worker actors. Defaults to the number of visible CUDA devices.

```json
{
  "implementation": {
    "backend": "torch",
    "torch_device": "cuda",
    "torch_batch_size": 4096
  }
}
```

Setting any `torch_*` option while `backend` is `"numpy"` produces a config warning that the option is ignored, rather than letting it look effective.

The three most commonly overridden options are also command-line flags on both `isofit apply_oe` and `isofit analytical_line`:

```
$ isofit analytical_line ... --backend torch --torch_device cuda --torch_batch_size 4096
```

The command line wins over the config file; omitting a flag means "use whatever the config says".

## Worker topology

The numpy backend runs one ray actor per core, so `n_cores` is the parallelism knob. The torch backend does not work that way.

Backend | Actors | Sized by
-|-|-
`numpy` | One per core | `n_cores`
`torch` on `cuda` | **One per GPU**, requested with `.options(num_gpus=1)` | `torch_num_gpu_workers`, defaulting to `torch.cuda.device_count()`
`torch` on `cpu` | One per core | `n_cores`

**`n_cores` does not control the number of GPU workers.** Handing every CPU core its own CUDA context would thrash the device and exhaust VRAM: each context carries its own copy of the resident LUT plus allocator headroom, and the contexts serialize against each other on the same SMs. One actor per GPU, fed with large batches, keeps the device saturated with a single copy of the state. Setting `n_cores` alongside `backend: torch` emits a config warning to that effect; use `torch_num_gpu_workers` instead.

`n_cores` still matters for the CPU-side work in the same job -- atmospheric interpolation, segmentation, and the ray CPU pool -- so it is not ignored, just not the GPU knob.

On the GPU, parallelism comes from the batch size instead. `torch_batch_size: auto` takes 70% of free device memory to leave allocator headroom, divides by an estimated per-pixel cost, and rounds down to a multiple of 64.

That per-pixel estimate is `8 * (4*ns^2 + 3*nw^2)` bytes, which is a deliberate **upper bound** rather than a derivation. The measured peak via `torch.cuda.max_memory_allocated` on an A100 at `nw=285`, `ns=425` is **6.78 MiB/pixel**, stable to 0.2% from batch 128 to 1024; the formula yields 7.37. Erring high means `auto` under-commits slightly rather than exhausting VRAM part-way through a batch, and it stays valid if one of the intermediates stops being computed in place. Re-measure rather than re-derive if the solve changes: deriving this figure from tensor bookkeeping has already produced answers wrong in both directions.

## Measured performance

The only measured figure available today is a **kernel-level** one:

Measurement | Value
-|-
`invert_analytical` solve kernel, batch 4096, A100, fp64 | 21,213 px/s
Same kernel, scalar per-pixel path, one CPU core | 96 px/s
Speedup vs. a single core | **221x**
Approximate speedup vs. a fully-loaded 32-core node | **~7x**

Read that table with its caveats attached:

* It measures **the solve kernel only**. It excludes I/O, atmospheric interpolation, the scalar initializer that seeds each pixel, ENVI writes, and ray overhead -- all of which are still on the CPU and none of which got faster.
* The 221x baseline is **a single core**. The honest comparison for a deployment is against a machine's full core count, which is where the ~7x figure comes from.
* fp64 throughput varies enormously across GPUs. T4, L4, and A10G class devices throttle fp64 by roughly 1:64 and must never be used to quote fp64 performance.

### Stage-level, measured

Amdahl's law applies with force to the kernel figure above: once the solve is
221x faster, the remaining scalar work dominates. The stage-level measurement
says by how much.

Scene: `image_cube/medium`, 1000 x 100 x 425 = 100,000 pixels, AVIRIS-NG,
multicomponent surface, `--presolve --segmentation_size 400
--pressure_elevation --analytical_line`.

Backend | Hardware | Stage time | Throughput
-|-|-|-
`numpy` | 8 CPU cores | 1139.8 s | 88 px/s
`torch` | 1x A100-40GB | 170.8 s | 585 px/s
speedup | | **6.67x** |

This is the `analytical_line` stage only, invoked directly, with the LUT,
presolve, OE and atmospheric interpolation already built. It is not a
whole-pipeline number.

The gap between 221x and 6.67x is the scalar remainder: geometry construction,
the algebraic initializer, ENVI reads and writes, and ray overhead. Those are
the next targets, not the solve.

### Parity on that run

Reflectance agreement between the two backends over the 37,995,763 pixel-bands
where both produced a physically plausible value:

Metric | Value
-|-
max abs difference | 5.96e-08
p99.9 abs difference | 1.49e-08
median abs difference | 0.0
mean abs difference | 4.20e-11
within 1e-4 | 100.000 %

The output cube is float32, whose machine epsilon is ~1.2e-7, so a maximum
absolute difference of 5.96e-08 is below one ULP: the two paths agree to the
limit the output format can represent, and half the values are bit-identical.
The larger *relative* difference (6.3e-3 max) occurs only where reflectance is
near zero and the denominator vanishes; the absolute error there is unchanged.

Note that this parity is against the CPU path **as it behaves today**, which is
the backend's contract. Where that path has a suspected defect -- see
`whiten_innovation` and its `strict_parity` flag in
`isofit/backends/torch/linalg.py` -- the batched backend reproduces the defect
deliberately rather than silently diverging.

## Benchmarking

`scripts/benchmarks/benchmark_backend.py` runs the `analytical_line` stage across backends on one fixed scene and writes a versioned JSON artifact.

```
$ python scripts/benchmarks/benchmark_backend.py \
    RDN_FILE LOC_FILE OBS_FILE ISOFIT_DIR \
    --isofit_config CONFIG.json \
    --backends numpy,torch \
    --torch_device cuda \
    --torch_batch_size 4096 \
    --repeats 3 \
    --output_dir bench_out \
    --output_json benchmark_result.json
```

What the harness does, and why each part matters for reproducibility:

* **Both backends run on the same host, in the same session.** BLAS build, clock speed, thermal state, and page cache all move the absolute numbers; running the legs back to back is what makes their ratio meaningful. Numbers from different machines are not comparable.
* **One warmup run per backend is excluded from timing.** It absorbs the numba JIT, the CUDA context creation, the LUT upload, and a cold page cache.
* **`--repeats` (default 3) timed runs, reported as the median**, so a single noisy repeat cannot move the headline.
* **Atmospheric interpolation is excluded.** Every run shares one `--atm_file`, created by the warmup. It is numpy-only preprocessing and identical for both backends.
* **Each run is a subprocess**, so it gets a clean ray cluster, a clean CUDA context, and a real exit code.
* **Every run is validated before anything is reported.** A nonzero exit, a timeout, a missing or wrong-shaped output cube, or a cube whose reflectance is not physically plausible aborts the benchmark. This matters more than it sounds: `initialize_output` pre-fills the output with zeros, so a run whose workers all died still leaves a complete, correctly-headed ENVI cube on disk -- and it finishes *fast*. The harness requires that most valid values fall in the open interval (0, 1), which is what distinguishes a retrieval from a pre-filled cube. It also diffs each backend's cube against the reference backend's, because a backend that is fast because it solved a different problem is not a speedup.

The artifact carries a `schema_version` plus an environment block -- torch version, resolved device name, dtype, batch size, `n_cores`, git SHA and dirty flag, and `ISOFIT_DEBUG` -- so a result file stays interpretable without the shell history that produced it. Note that `ISOFIT_DEBUG=1` bypasses ray and forces the run serial; the harness records it and warns, but timings taken that way do not describe a parallel deployment.

CUDA benchmarking from a non-CUDA development machine can go through `scripts/modal/isofit_gpu.py`, which mounts the working tree onto a rented A100 or H100:

```
$ modal run scripts/modal/isofit_gpu.py::benchmark --gpu a100
```

## Parity bands

Correctness is gated by parity tests that compare each torch module against the existing numpy implementation on the same inputs. The tolerances widen down the stack, because each layer composes the error of the ones beneath it:

Test module | Band
-|-
`test_torch_lut.py` | rtol 1e-13 (fp64), 3e-6 (fp32)
`test_torch_instrument.py` | rtol 1e-12
`test_torch_forward.py` | rtol 1e-12
`test_torch_surface.py` | rtol 1e-11
`test_torch_seps.py` | rtol 1e-11
`test_torch_linalg.py` | rtol 1e-8
`test_torch_analytical.py` | rtol 1e-7, atol 1e-9

The end-to-end analytical solve sits at the loose end of that range because it is the composition of every layer above plus two Cholesky stages, and because the batched implementation legitimately reorders floating-point operations relative to the scalar loop.

These tests run on CPU and are safe to run anywhere. See [Testing](testing.md#torch-backend-parity-tests) for the markers and how to run them.
