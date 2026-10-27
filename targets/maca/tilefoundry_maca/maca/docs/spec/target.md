# MACA analysis target

> Analysis-only. This target prices MACA programs; it does not emit CUDA,
> MACA C, or TileLang. Implementations live in mcDeepGemm's MACA-C kernels or
> in mcTileLang.

## 1. Why a separate target

TileFoundry's shipped targets are `nvidia` (CUDA codegen), `apple` (AMX
codegen) and `maca` (analysis). MACA is not CUDA at the instruction level: a
wave is 64 lanes, the execution unit is an AP with four PEUs, the compiler is
`mxcc`/`cucc`, and the scalar / special / conversion instruction rates differ
from an SM. Pricing a MACA program with the H200 target gives the right
*logical* traffic but the wrong rates.

`targets/maca` reuses the CUDA **fact** schema (`tilefoundry.cuda.device/v3`,
`tilefoundry.cuda.architecture/v3`) and the CUDA target's fact projections, and
installs MACA hardware documents. The topology vocabulary is therefore
`gpu` / `cta` / `thread`, with `cta` = AP (`28` or `32` instances) and
`thread` = one 64-lane wave; the documents state `max_threads_per_warp = 64`.

No codegen is claimed: `MacaTarget` inherits only the analysis surface. Do not
pass it to `compile`/`scaffold` expecting CUDA or MACA C output.

## 2. Products and facts

| product | arch id | APs | L2 | HBM model | BF16 | FP8 |
|---|---|---:|---:|---:|---:|---:|
| `maca.c500` | `maca.xcore1000` | 104 | 8 MiB | 1490.5 GB/s measured read+write (nominal 1843.2) | 240 TFLOPS | unavailable |
| `maca.c600` | `maca.xcore1500` | 32 | 2 MiB | 1000.6 GB/s measured read+write (nominal 1638) | 193.35 TFLOPS | unavailable |
| `maca.c600u` | `maca.xcore1600` | 28 | 2 MiB | 1482.0 GB/s measured read+write (nominal 2048) | 219.05 TFLOPS | 438.1 TFLOPS |
| `maca.c600ul` | `maca.xcore1600` | 32 | 2 MiB | 1482.0 GB/s (estimate from C600U; nominal 2048) | 275.25 TFLOPS | 550.5 TFLOPS |

Architecture documents also state: `max_threads_per_cta = 2048`,
`max_threads_per_warp = 64`, `max_warps_per_cta = 32`,
`max_resident_ctas_per_sm = 8`, `shared_memory_per_sm/per_cta` (64 KiB on
XCORE1000, 128 KiB on XCORE1500/1600), `registers_per_sm_32bit = 131072`,
`rmem_owner = thread`, and no separate tensor memory.

The bandwidth column is the measured read+write envelope (`torch-add`,
L2-flushed linear fit) from
`mcDeepGemm/deep_gemm/tests/perf_data/bandwidth/20260818_MetaX_*`; read-only
ceilings are higher (C500 1624, C600 979.9, C600U 1628.5). The compute peaks
are nominal `maca-perf-model` values; each fact's `conditions` states the
measured bare-MMA ceiling where one exists (C600U BF16 215.5/219.05, FP8
415.7/438.1). INT8 peaks in the table are documentation only: the schema has
no `int8` throughput key and the IR has no `i8` dtype, so the analysis prices
integer work through `service.integer`, not a tensor-core peak.

Sources: the workspace `maca-perf-model` skill (AP/L2/nominal HBM/MMA peaks),
`mcDeepGemm`'s `perf_data/bandwidth/20260818_MetaX_*` (measured envelopes),
`mcDeepGemm/docs/mega_mhc_maca_c_occupancy_20261017.md` (C600U registers, smem,
wave, occupancy), and `torch.cuda.get_device_properties(0)` on an MXC600-U
(28 APs, 131072 regs/AP, 131072 B smem/AP, 2048 threads/AP).

Every fact states `origin` (`vendor`, `measured`, `reference`, `derived`,
`estimated`) and `conditions`. On `maca.c600u` the 1900 MHz XCORE clock
(`mx-smi --show-clock`) and the 800 ns HBM latency (pointer-chase: 769-822 ns at
1-4 chains/AP, L2-resident 308-323 ns) are measured. The C500/C600/C600UL
clocks and latencies, and the 128 B/clock/AP shared-memory rate, remain
estimates.

## 3. What the analysis reads

`MacaTarget.get_facts` inherits the CUDA projections:

- `TopologyFacts`: `gpu` (device_count, default 1), `cta` (AP count), `thread`
  (APs × 2048).
- `MemoryHierarchyFacts`: explicit `gmem`, `smem`, `rmem`; implicit `l1`, `l2`;
  `l2` caches `gmem`, `l1` caches `l2`, `l1` shares capacity with `smem`.
- `ThroughputFacts`: dense device FLOP/s per dtype and HBM bytes/s.
- `PerformanceServiceFacts`: per-level float/service/shared-memory rates; the
  CTA-level gmem rate is capped by Little's law using the stated gmem latency
  and the HIR's in-flight bytes.

The `compute-cost` family is target-independent; `memory` and `roofline` read
the target. A program whose workload needs a dtype the product does not list
(for example FP8 on C500/C600) has no peak for that dtype by construction.

## 4. Using it

The target is discoverable through the same agent-facing surfaces as the
shipped ones:

```sh
tilefoundry target list          # MacaTarget("maca.c500|c600|c600u|c600ul")
tilefoundry target show maca.c600u
tilefoundry spec maca/target     # this page
```

In an authored HIR:

```python
from tilefoundry_maca import MacaTarget
_TARGET = MacaTarget("maca.c600u")

@module(entry="s3", target=_TARGET, topologies=(...))
class S3: ...
```

then `tilefoundry analyze model.py:S3 out.txt --compute-cost --memory --roofline`.

## 5. MACA-C programming model

The mapping from a hand-written MACA-C kernel to the analysis vocabulary:

| MACA-C concept | HIR |
|---|---|
| CTA | `cta` topology, one instance per AP |
| wave (64 lanes) | `thread` topology (`max_threads_per_warp = 64`) |
| PEUs inside an AP | not a topology axis; read occupancy from the documented `max_warps_per_cta` / `max_resident_ctas_per_sm` facts and `mxcc --resource-usage` |
| registers (`MTregs` per thread) | `rmem`, owner `thread`; per-AP budget `registers_per_sm_32bit = 131072` |
| shared memory | `smem`; 128 KiB/AP on C600U, 64 KiB on C500 |
| MMA / FP32 / conversion throughput | `ThroughputFacts` and `PerformanceServiceFacts` |

Workflow:

1. **Mirror the kernel's tiles.** Give each tensor the CTA and thread layout
   the kernel actually uses. A tensor with no CTA axis is **replicated per
   CTA**: that is how a weight re-read is charged. `@logical` is the unique
   bytes; `@total` multiplies by the repeats (CTAs, waves, splits).
2. **Price both sides.** Run `--compute-cost --memory --roofline`; `bound-by`
   names the binding side, and the per-op traffic lines show where the bytes
   go.
3. **Validate the absolute number.** The roofline also depends on the layout's
   access granularity (28 contiguous elements per thread prices differently
   from the kernel's 16-byte packets). Use the model for relative comparisons
   (e.g. one tiling versus another) and the target facts plus measurements for
   absolutes: `mxcc --resource-usage`, `mx-smi --show-clock`, and a streaming
   ceiling. A placement advisory (`rmem placement peak ...`) is coarse when the
   HIR mesh is coarser than the per-thread tile; occupancy still comes from
   `mxcc`.

Worked example — mega_mhc S1 (`tilefoundry_maca/maca/models/mega_mhc_s1.py`):

```sh
tilefoundry analyze targets/maca/tilefoundry_maca/maca/models/mega_mhc_s1.py:S1MixGemm \
    /tmp/s1.txt --compute-cost --memory --roofline
```

At `BM=128` the model reports `gmem:r2.84GiB/w2.19GiB@total`; 672 MiB of the
read is the per-CTA `fn` re-read (256 CTAs × 2.62 MiB). At `BM=64` the total
moves from 5.03 GiB to 5.70 GiB (+13.3%), matching the measured penalty of the
ct64 arm and showing why it kept 128 tokens per CTA. The same method on the S3
tail gives the calibrated 2.11 ms memory floor and the instruction-bound 3.6 ms
reference (section 7).

## 6. Python 3.10

The core tree and both target distributions declare `requires-python >= 3.10`.
The compatibility changes are:

| 3.11/3.12 construct | 3.10 replacement |
|---|---|
| 15 PEP 695 generic declarations (`def f[T]`, `class C[T]`) | module-level `TypeVar` + `Generic[T]` |
| `type ShapeDim = ...` | `ShapeDim: TypeAlias = ...` |
| `enum.StrEnum` | `tilefoundry/_compat.py` str-Enum shim |
| `tomllib` | `_compat.py` try/except with `tomli` |
| `Tuple[is_ranked_tensor()]` (a pattern as a type argument) | `tuple_annotation(...)` (`types.GenericAlias`) |
| `events: ClassVar` with `from __future__ import annotations` | `events: ClassVar[tuple[str, ...]]` |

The `ClassVar` change is not cosmetic: with postponed annotations, 3.10's
`get_type_hints` evaluates the string `'ClassVar'` and raises
`TypeError: Plain typing.ClassVar is not valid as type argument` when the
report renders a record. The four record fields are in
`analysis/liveness.py` and `analysis/metadata.py`.

Install in the py310 MACA environment (validated with
`tilefoundry analyze ... --compute-cost --memory --roofline` on the mega_mhc S3
and S1 HIRs):

```bash
python -m pip install isl-python==0.1.9 ortools==9.9.3963 graphviz \
    pandas pytz tzdata absl-py immutabledict protobuf
python -m pip install "setuptools-scm==8.1.0"
python -m pip install -e /path/to/TileFoundry --no-deps --no-build-isolation
python -m pip install -e /path/to/TileFoundry/targets/nvidia --no-deps --no-build-isolation
python -m pip install -e /path/to/TileFoundry/targets/maca --no-deps --no-build-isolation
tilefoundry spec maca/target
```

The internal index currently has `isl-python` 0.1.9 and `ortools` 9.9 for
cp310, below the py312 versions; `analyze` works with them. `ortools` imports
`pandas` from `cp_model`, and `setuptools-scm` 10.x needs `vcs_versioning`, so
8.1.0 is used for the editable builds. `transformers` is only needed by
`models`/orchestrator, not by `analyze`.

## 7. Multi-wave footprint accounting

The S3 HIR in
`mcDeepGemm/docs/experiments/20261026_c600u_mega_mhc_tilelang_s3/s3_fused.py`
uses 128 tokens per CTA and a 1024-wide thread mesh. Its grid has 256 CTAs on a
28-AP target, i.e. `ceil(256/28) = 10` waves.

An earlier release reported `roofline ideal-ns=15744188` (15.7 ms) for this
HIR. That was a **footprint-to-roofline bug**, not a hardware floor:

- `footprint.reached_by` could not restrict the access to one 28-unit wave
  (the mesh position was unavailable), but it still returned the whole-grid
  footprint (1.12 GiB) instead of `None`.
- `analysis/memory.py` then attached that footprint with
  `windows = ceil(256/28) = 10`, so `Footprint.bytes_at` reported
  `logical = 1.12 GiB`, `total = 11.21 GiB`.
- `analysis/roofline.py` uses `total`, giving 11.21 GiB / 764.6 GB/s = 15.7 ms.

The fix is in `src/tilefoundry/analysis/footprint.py`: when the wave
restriction cannot be applied, `reached_by` returns `None` and
`moving_boundaries` marks `wave_stated = False`. The roofline then falls back to
the call's traffic record. With the original 600 ns estimate `maca.c600u`
reported `ideal-ns=1579201` (1.58 ms = 1.12 GiB / 764.6 GB/s). After the
pointer-chase (`probe_latency.py`: 800 ns) and clock (`mx-smi`: 1900 MHz)
calibration it reports `ideal-ns=2105600` (**2.11 ms** = 1.12 GiB / 573.4
GB/s). The H200 tutorial's one-wave cases are unchanged (4952 / 8697 ns).

The cta-level rate is 573.4 GB/s because the target's Little's-law cap uses
`inflight = 16 KiB/CTA`, `active = min(256, 28) = 28` and the measured 800 ns
latency: `16384 × 28 / 800 ns = 573.4 GB/s`. The pure HBM floor at the measured
read+write envelope is 1.12 GiB / 1482 GB/s = 0.81 ms (1.02 ms if the
mega_mhc copy-probe ceiling of 1183 GB/s is used). The measured MACA-C S3 is 3.6 ms, above both, so that
stage is instruction-bound.

Two placement advisories remain and are still meaningful:

- `rmem placement peak 7.11MiB exceeds capacity 512.00KiB` — the HIR's
  1024-wide thread mesh is coarser than the real kernel's per-thread tile.
- the L2 advisory still reports the whole-grid hold (1.12 GiB) against the
  2 MiB cache; it is a conservative warning, not part of the roofline.

For absolute gates, make the HIR physically placeable (per-CTA rmem ≤ 512 KiB,
per-wave L2 window ≤ 2 MiB) and calibrate the estimated latency/clock. Until
then, use 0.81-2.11 ms as the memory floor (1.02-2.11 ms with the copy-probe
ceiling) and the measured 3.6 ms as the current instruction-bound reference.
