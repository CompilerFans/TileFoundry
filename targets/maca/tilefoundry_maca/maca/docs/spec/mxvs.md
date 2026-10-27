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

The bundled package metadata reports `Version: 0.0.1`. This build's quirks:

- `--bref` is required for `memory benchmark`: without it the tool queries
  MetaXLink and fails with `MxSmlFailure: 1` on this host.
- `--dies` requires `--detail`, otherwise the command prints a usage error.
- `ops` without `--blas` reports `-0.00` on this build; use `--blas`.
- `ops --models` accepts `int8, bf16, fp16_matrix, fp32_matrix, tf32`; there is
  no `fp8` model (the C600U FP8 ceiling needs the mcDeepGemm bare-MMA probe).
- A multi-model `ops` run (or an unpinned width) can aggregate **both dies**
  and print BDF `0000:25:00.0,0000:24:00.0` with doubled values. Check the BDF
  column (or the JSON `bdf` field) before recording a number; per-die values
  below used BDF `0000:24:00.0`.

## 2. C600U measurements (2026-10-27, GPU0, die 0)

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

## 3. Reading the numbers

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
  `perf_data` torch-add fit (1005 vs 1482 GB/s). Never compare across tools or
  modes; quote the tool, mode and size.
- mxvs `ops --blas` is GEMM throughput (memory and epilogue included), so it
  sits below the bare-MMA probes. The target's roofline still uses the design
  peak; the measured values answer "what did this machine reach, measured by
  this tool".
- The MMA issue rates come in pairs: **BF16 = FP16** (16x16x16; 219.05 on
  C600U) and, on xcore1600, **INT8 = FP8** (16x16x32; 438.1). A pair shares
  its design peak and its bare-MMA measurement. mxvs reads bf16 196.48 and
  fp16 196.53 (GEMM-achieved, the same rate within noise); it has no fp8
  model, so its int8 row (371 TOPS, GEMM-achieved) must not be compared with
  the FP8 bare-MMA probe (415.7) as if the rated throughputs differed.
- The target facts carry these readings in their `conditions` (HBM bandwidth,
  BF16/FP16, FP8) so `tilefoundry target show maca.c600u` shows both the
  design value and the evidence.

## 4. Feed a new measurement back

1. Run the command with a pinned die and a recorded size/mode.
2. Keep the JSON (`--json`) next to the run; quote the BDF and `datasize` in
   the record.
3. Add the number to the fact's `conditions` as measured evidence (never
   replace the design value) and, if it is a new caliber, extend this page.
