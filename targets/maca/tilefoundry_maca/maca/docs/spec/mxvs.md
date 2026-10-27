# mxvs measurements for the MACA target

`mxvs` (MetaX Validation Suite) is the vendor benchmark tool shipped with
MACA. This page records how to run it on a C600U, what the 2026-10-27 run
measured, and how those numbers relate to the target's **design** facts and to
the other measured calibers. The rule is the same as the rest of the target:
a fact carries the design value; every measured number is evidence with its
tool, mode and size attached.

## 1. Running the AppImage

The shipped `/usr/bin/mxvs` is an AppImage that needs `libfuse.so.2`. On a host
without FUSE, extract it once (the bundle has its own python3.8 and libffi):

```sh
mkdir -p /tmp/mxvs && cd /tmp/mxvs
/usr/bin/mxvs --appimage-extract            # creates squashfs-root/
export APPIMAGE=$PWD/squashfs-root/AppRun   # AppRun's own fallback miscounts
export APPDIR=$PWD/squashfs-root            # one directory level, so set both
APP=$APPDIR/AppRun
```

The bundled package metadata reports `Version: 0.0.1` on both hosts measured
below, but behavior is host/build dependent, so each measurement section
records its own quirks. Common to both:

- `--dies` requires `--detail`, otherwise the command prints a usage error.
- `ops --models` never accepts `fp8`; the FP8 ceiling needs the mcDeepGemm
  bare-MMA probe.
- On the C600U host a multi-model `ops` run (or an unpinned width) aggregated
  **both dies** and printed BDF `0000:25:00.0,0000:24:00.0` with doubled
  values; on C600 a pinned `-d 0 --dies 0` run stayed on one BDF. Always check
  the BDF column (or the JSON `bdf` field) and `datasize(Byte)`; the per-die
  values below are single-BDF readings.

## 2. C600U measurements (2026-10-27, GPU0, die 0)

On that host `--bref` was required for `memory benchmark` (without it the
MetaXLink query fails with `MxSmlFailure: 1`), `ops` without `--blas` reported
`-0.00`, and `ops --models` accepted `int8, bf16, fp16_matrix, fp32_matrix,
tf32`. Every JSON `bdf` below is `0000:24:00.0`.

Bandwidth, `memory benchmark -d 0 --dies 0 --detail`:

| size | read-only GB/s | kernel-copy GB/s |
|---:|---:|---:|
| 64 MB | 913.4 | 821.9 |
| 256 MB | 1339.8 | 963.4 |
| 1 GB | 1493.9 | 1003.5 |
| 4 GB | 1547.6 | 1015.9 |
| 10 GB | 1533.6 | 1005.5 |

Two 10 GB runs gave read `1510.3 / 1533.6` and copy `1002.9 / 1005.5`; the
read plateau is ~1510-1550 GB/s, the copy plateau ~1003-1016 GB/s. Raw JSON
for every row (bandwidth sweep and per-die `ops`) is in
`docs/mxvs/20261027_c600u/` next to this page; check each file's `bdf` and
`datasize(Byte)` before quoting it.

Compute, `ops -d 0 --models <model> --blas --detail` (GEMM-achieved, not a
bare-MMA ceiling):

| model | mxvs TOPS/TFLOPS |
|---|---:|
| int8 | 371.17 |
| bf16 | 196.48 |
| fp16_matrix | 196.53 |
| tf32 | 86.24 |
| fp32_matrix | 2.43 |

`fp32_matrix` is the BLAS path and is not a meaningful peak here; `tf32` has
no design fact in the schema (recorded as evidence only).

## 3. C600 measurements (2026-10-27, MXC600-A GPU0, die 0)

Host `MXC600-A`: two MetaX C600-A boards, 32 AP each (`mx-smi
--show-ap-usage` shows 4 dpc x 8), HBM3e 72 GB decimal, xcore1500. Software:
MACA 3.0.1436.20261017, KMD 3.9.6, VBIOS 2.3.3.0, mx-smi 2.3.4. Tool:
`/usr/bin/mxvs -> /opt/maca-3.5.3.17-20261027/bin/mxvs`, package
`mxvs 0.0.1`. `mxvs devices` reports GPU0 `MetaX C600-A` BDF
`0000:07:00.0` (measured) and GPU1 `0000:47:00.0` (idle, not used).
`mx-smi --show-clock` reads XCORE 504 MHz idle and 1475 MHz during an mxvs
run, MC 3200 MHz.

Commands (default iterations: 10000 for `memory benchmark`, 100 for
`ops --blas`):

```sh
mxvs memory benchmark -d 0 --dies 0 --detail -s 10GB --json out.json
mxvs memory benchmark -d 0 --dies 0 --detail --kernel-copy -s 10GB --json out.json
mxvs ops -d 0 --models bf16 --blas --detail --json out.json
```

Quirks observed on this host (same package version as C600U, different
behavior):

- `--bref` is **not** required: GPU info displays and both `memory benchmark`
  and `ops` complete without it. bf16 reads 185.79 without `--bref` and 185.80
  with it, so the flag does not change the number.
- `--dies` still requires `--detail` (`Error: Parameter --detail must be
  provided when --dies is set`).
- The non-blas `ops` path **works** here (the C600U host reported `-0.00`). It
  reads ~99.8% of nominal (ops table below), i.e. it behaves like a benchmark
  loop at the MMA issue rate; it is not a calibrated bare-MMA probe.
- `ops --models` accepts only `int8, bf16, fp16_matrix, tf32`: it rejects both
  `fp8` and `fp32_matrix` (`Error: Invalid value: ...`). The C600U build
  accepted `fp32_matrix`, so the two shipped bundles differ despite both
  reporting `mxvs 0.0.1`.
- A pinned `-d 0 --dies 0` run stays on one BDF. Every JSON in
  `docs/mxvs/20261027_c600/` carries `"bdf": "0000:07:00.0"` and the requested
  `datasize(Byte)` (67108864, 268435456, 1073741824, 4294967296, 10737418240
  for 64 MB-10 GB), so no both-dies aggregation occurred.

Bandwidth (GB/s):

| size | read-only | kernel-copy |
|---:|---:|---:|
| 64 MB | 971.6 (951.3 in an earlier no-bref probe) | 830.3 |
| 256 MB | 1054.5 | 851.4 |
| 1 GB | 1057.3 | 855.5 |
| 4 GB | 1059.5 | 856.6 |
| 10 GB | 1063.3 / 1062.0 | 856.4 / 856.5 |

The 10 GB row is two runs. Read plateaus at ~1055-1063 GB/s (the 64 MB point
is 8.6% lower); kernel-copy plateaus at ~856 GB/s.

Compute, `ops -d 0 --models <model> --detail` with and without `--blas`
(default iterations):

| model | `--blas` GEMM TOPS/TFLOPS | non-blas ops | nominal |
|---|---:|---:|---:|
| int8 | 365.73 | 385.86 | 386.7 (documented INT8 rate) |
| bf16 | 185.80 | 192.97 | 193.35 |
| fp16_matrix | 185.79 | 192.95 | 193.35 |
| tf32 | 80.22 | 96.46 | — |

Design vs measured (the target rule: measured evidence never replaces a
design value):

| metric | design (fact) | mxvs 0.0.1 (die 0) | other measured evidence |
|---|---:|---:|---|
| HBM read 10 GB | 1638 GB/s | 1062-1063 | 979.9 (perf_data read-only fit) |
| HBM read+write 10 GB | 1638 GB/s | 856.4-856.5 | 1000.6 (torch-add fit); 829.1 (copy fit) |
| INT8 GEMM | 386.7 TOPS (documented) | 365.73 / 385.86 | — |
| BF16 GEMM | 193.35 TFLOPS | 185.80 / 192.97 | — |
| FP16 GEMM | 193.35 TFLOPS | 185.79 / 192.95 | — |
| FP8 GEMM | unavailable (emulated by BF16) | not supported | — |
| TF32 GEMM | — | 80.22 / 96.46 | — |

Unlike C600U, the mxvs read-only number here (1063 GB/s) is **above** the
perf_data read-only fit (979.9 GB/s); on C600U it was below (1510-1534 vs
1628.5). The two are different access patterns, so neither corrects the other.

## 4. Reading the numbers

C600U design vs measured:

| metric | design (fact) | mxvs 0.0.1 (die 0) | other measured evidence |
|---|---:|---:|---|
| HBM read 10 GB | 2048 GB/s | 1510-1534 GB/s | 1628.5 (perf_data read-only fit) |
| HBM read+write 10 GB | 2048 GB/s | 1003-1006 GB/s | 1482.0 (torch-add fit); 1183 (mega_mhc copy probe) |
| INT8 GEMM | 438.1 TOPS | 371.17 | — |
| BF16 GEMM | 219.05 TFLOPS | 196.48 | 215.5 (bare-MMA probe) |
| FP16 GEMM | 219.05 TFLOPS | 196.53 | — |
| FP8 GEMM | 438.1 TFLOPS | not supported | 415.7 (bare-MMA probe) |
| TF32 GEMM | — | 86.24 | — |

- mxvs `memory benchmark` read-only is a different access pattern from the
  `perf_data` read-only fit; mxvs `--kernel-copy` is lower than the
  `perf_data` torch-add fit (1005 vs 1482 GB/s on C600U). Never compare across
  tools or modes; quote the tool, mode and size.
- mxvs `ops --blas` is GEMM throughput (memory and epilogue included), so it
  sits below the bare-MMA probes. The target's roofline still uses the design
  peak; the measured values answer "what did this machine reach, measured by
  this tool".
- The non-blas `ops` arm is host/build dependent: it works on the C600 host
  (192.97 bf16 / 192.95 fp16 / 385.86 int8, ~99.8% of nominal) and reported
  `-0.00` on the C600U host. Never mix it with `--blas` GEMM readings, and do
  not call it a bare-MMA probe.
- The MMA issue rates come in pairs: **BF16 = FP16** (16x16x16; 219.05 on
  C600U, 193.35 on C600) and, on xcore1600, **INT8 = FP8** (16x16x32; 438.1).
  A pair shares its design peak and its bare-MMA measurement. mxvs reads bf16
  196.48 / fp16 196.53 on C600U and 185.80 / 185.79 on C600 (GEMM-achieved,
  the same rate within noise); it has no fp8 model, so its int8 row (371.17
  TOPS on C600U, 365.73 on C600, GEMM-achieved) must not be compared with the
  FP8 bare-MMA probe (415.7) as if the rated throughputs differed.
- The target facts carry these readings in their `conditions` (HBM bandwidth,
  BF16/FP16, FP8) so `tilefoundry target show maca.c600u` and
  `tilefoundry target show maca.c600` show both the design value and the
  evidence.

## 5. Feed a new measurement back

1. Run the command with a pinned die and a recorded size/mode.
2. Keep the JSON (`--json`) in `docs/mxvs/<yyyymmdd>_<device>/` (the package
   data includes `docs/mxvs/*/*.json`); quote the BDF and `datasize` in the
   record.
3. Add the number to the fact's `conditions` as measured evidence (never
   replace the design value) and, if it is a new caliber, extend this page and
   the measured-evidence table in `targets/maca/README.md`.
