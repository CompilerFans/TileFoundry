# TileFoundry MACA targets

Analysis-only targets for MetaX MACA devices (C500 / C600 / C600U / C600UL).
They reuse TileFoundry's CUDA **fact** schema and projections (topology,
memory hierarchy, compute/service rates) and install MACA hardware documents,
so `tilefoundry analyze` can price a MACA-C or mcTileLang kernel against MACA
numbers. **There is no CUDA code generation and no MACA C code generation in
this package**; kernels are implemented with mcTileLang or hand-written MACA C.

The agent-facing page is a spec, not a wiki:

```sh
pip install "tilefoundry[maca]"          # or: pip install -e targets/maca --no-deps
tilefoundry spec maca/target             # sections: products, model, workflow, caveats
tilefoundry target list                  # MacaTarget("maca.c500|c600|c600u|c600ul")
tilefoundry target show maca.c600u       # every fact and its evidence
```

## Products

| id | arch | APs | L2 | HBM design | HBM measured (rw / read) | BF16/FP16 design | FP8 design | INT8 (doc; = FP8 rate) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `maca.c500` | `maca.xcore1000` | 104 | 8 MiB | 1843.2 GB/s | 1490.5 / 1624 GB/s | 240 TFLOPS | unavailable (emulated) | 480 TOPS |
| `maca.c600` | `maca.xcore1500` | 32 | 2 MiB | 1638 GB/s | 1000.6 / 979.9 GB/s | 193.35 TFLOPS | unavailable (emulated) | 386.7 TOPS |
| `maca.c600u` | `maca.xcore1600` | 28 | 2 MiB | 2048 GB/s | 1482.0 / 1628.5 GB/s | 219.05 TFLOPS | 438.1 TFLOPS | 438.1 TOPS |
| `maca.c600ul` | `maca.xcore1600` | 32 | 2 MiB | 2048 GB/s | 1482.0 est / — | 275.25 TFLOPS | 550.5 TFLOPS | 550.5 TOPS |

Every fact carries `origin` (`vendor`, `measured`, `reference`, `derived`,
`estimated`) and `conditions`. Following the NVIDIA target's convention, the
**design** columns are the vendor nominal values and are what the roofline
uses. Measured values are evidence, recorded in each fact's `conditions` (and
the measured column above), never substituted into a design fact:

- HBM: read+write / read-only envelopes from
  `mcDeepGemm/deep_gemm/tests/perf_data/bandwidth/20260818_MetaX_*`
  (L2-flushed fits); the mega_mhc copy probe (1183 GB/s) is one more measured
  point, from a session with a ~24% speed-mode flip.
- MMA: C600U design BF16 219.05 / FP8 438.1 TFLOPS; measured bare-MMA ceilings
  215.5 / 415.7. C500 design INT8 480 TOPS; measured 477.9. No C600 bare-MMA
  probe is recorded.
- Machine state: C600U 1900 MHz (`mx-smi --show-clock`) and 800 ns HBM latency
  (pointer chase) are measured; C600UL inherits them as estimates; C500/C600
  clocks and latencies remain estimates.

INT8 peaks are documented for reference only: the CUDA fact schema has no
`int8` throughput key and the IR has no `i8` dtype, so the analysis prices
integer work through the scalar `service.integer` fact, not a tensor-core
peak. The MMA issue rates come in pairs, and a pair shares its design peak
and its bare-MMA measurement:

- **INT8 = FP8** (16x16x32 MMA): 438.1 on C600U, 550.5 on C600UL. mxvs has no
  fp8 model; its int8 arm is a GEMM (371 TOPS achieved), not a different rate
  from the FP8 bare-MMA probe (415.7).
- **BF16 = FP16** (16x16x16 MMA): 219.05 on C600U, 275.25 on C600UL. mxvs
  reads them as 196.48 (bf16) and 196.53 (fp16) GEMM-achieved, and the
  bare-MMA probe is 215.5 for both.

C500/C600 have no native FP8 (emulated by BF16), so the INT8=FP8 pair applies
only to xcore1600.

Measured evidence, by tool and mode (never mixed into the design facts):

| metric | C500 | C600 | C600U | C600UL |
|---|---:|---:|---:|---:|
| HBM read-only, perf_data fit (GB/s) | 1624 | 979.9 | 1628.5 | — |
| HBM read+write, perf_data fit (GB/s) | 1490.5 | 1000.6 | 1482.0 | — |
| HBM read 10 GB, mxvs 0.0.1 (GB/s) | — | — | 1510-1534 | — |
| HBM kernel-copy 10 GB, mxvs (GB/s) | — | — | 1003-1006 | — |
| GEMM bf16, mxvs `--blas` (TFLOPS) | — | — | 196.48 | — |
| GEMM fp16, mxvs `--blas` (TFLOPS) | — | — | 196.53 | — |
| GEMM int8, mxvs `--blas` (TOPS) | — | — | 371 | — |
| GEMM tf32, mxvs `--blas` (TFLOPS) | — | — | 86.2 | — |
| bare-MMA bf16, probe (TFLOPS) | — | — | 215.5 | — |
| bare-MMA fp8, probe (TFLOPS) | — | — | 415.7 | — |

mxvs 0.0.1 has no fp8 model and its non-blas ops path reports -0.00 on this
build. The dtype pairs share their issue rate (INT8=FP8 on xcore1600,
BF16=FP16 everywhere), so compare a measurement only with the same tool and
mode; the extraction workaround,
the 64 MB-10 GB sweep and the per-die caveats are in
`tilefoundry spec maca/mxvs`; raw JSON is under `docs/mxvs/20261027_c600u/`.

## MACA-C programming model

| MACA-C concept | HIR |
|---|---|
| CTA | `cta` topology, one instance per AP (`28` on C600U) |
| wave (64 lanes) | `thread` topology; the hardware document states `max_threads_per_warp = 64` |
| PEs inside an AP | not a topology axis; express occupancy with the documented `max_warps_per_cta` / `max_resident_ctas_per_sm` facts |
| registers (`MTregs`) | `rmem`, owned by `thread`, capacity `registers_per_sm_32bit = 131072` per AP |
| shared memory | `smem`; 128 KiB/AP on C600U, 64 KiB on C500 |
| MMA peak / FP32 service | `ThroughputFacts` and `PerformanceServiceFacts` per dtype |

Workflow:

1. Write the HIR with the kernel's real tiles: the CTA tile over tokens, the
   wave layout over KV/heads, and the storage level of each operand. A tensor
   with **no CTA axis is replicated per CTA** — that is how a weight re-read is
   priced (see the example below).
2. `tilefoundry analyze model.py:Entry out.txt --compute-cost --memory --roofline`.
3. Read the per-op traffic lines, not only the summary: `@logical` is the
   unique bytes, `@total` multiplies by the executed repeats (per-CTA
   replication, waves, splits), and `bound-by` names the binding side.
4. Validate the absolute number against the machine: `mxcc --resource-usage`
   for registers/stack, `mx-smi --show-clock` for clocks, and a measured
   streaming ceiling for bandwidth. The roofline's absolute value also depends
   on the layout's access granularity (a 28-element-per-thread layout prices
   differently from the kernel's 16-byte packets); use the model for
   **relative** comparisons and the target facts for absolutes.

### Worked example: mega_mhc S1

`tilefoundry_maca/maca/models/mega_mhc_s1.py` is the residual-mix + prenorm-GEMM pass of
mcDeepGemm's `bf16_mega_mhc_maca_c.cuh`, at the traffic level:

```sh
tilefoundry analyze targets/maca/tilefoundry_maca/maca/models/mega_mhc_s1.py:S1MixGemm \
    /tmp/s1.txt --compute-cost --memory --roofline
```

With `BM=128` the model reports `gmem:r2.84GiB/w2.19GiB@total`, of which
**672 MiB is the per-CTA `fn` re-read** (256 CTAs x 2.62 MiB). Setting `BM=64`
(512 CTAs) moves the total from 5.03 GiB to 5.70 GiB (**+13.3%**), which is the
measured penalty of mcDeepGemm's ct64 arm; it is why the K=64 design had to
keep 128 tokens per CTA. The same workflow on the S3 tail
(`mcDeepGemm/docs/experiments/20261026_c600u_mega_mhc_tilelang_s3/`) gives the
calibrated 2.11 ms memory floor and the instruction-bound 3.6 ms reference.

## Boundaries

- Analysis only: `MacaTarget` inherits only the analysis surface; do not pass
  it to `compile` / `scaffold` expecting CUDA or MACA C output.
- Place-by-placement advisories (`rmem placement peak ...`) are coarse when the
  HIR thread mesh is coarser than the kernel's per-thread tile; they are
  warnings, and occupancy must still be read from `mxcc --resource-usage`.
- FP8 on C500/C600 has no peak by construction (the hardware document lists it
  as emulated); a workload that needs it has no denominator there.

## Python 3.10

The core tree and target distributions declare `requires-python >= 3.10` so the
target can live in the MACA environment (MACA torch and mcTileLang are
cp310-only). The compatibility shims (`_compat.py`, `Generic`/`TypeVar`,
`ClassVar[...]`, `tuple_annotation`) are listed in `tilefoundry spec maca/target`
section 6. Install in the py310 environment:

```bash
python -m pip install isl-python==0.1.9 ortools==9.9.3963 graphviz \
    pandas pytz tzdata absl-py immutabledict protobuf
python -m pip install "setuptools-scm==8.1.0"
python -m pip install -e /path/to/TileFoundry --no-deps --no-build-isolation
python -m pip install -e /path/to/TileFoundry/targets/nvidia --no-deps --no-build-isolation
python -m pip install -e /path/to/TileFoundry/targets/maca --no-deps --no-build-isolation
tilefoundry spec maca/target
```
