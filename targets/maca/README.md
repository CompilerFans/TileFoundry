# TileFoundry MACA analysis targets

Analysis-only targets for MetaX MACA devices. They reuse TileFoundry's CUDA
hardware-document schema and fact projections (topology, memory hierarchy,
compute/service rates) so `tilefoundry analyze` can price a program against MACA
numbers. **There is no CUDA code generation and no MACA C code generation in
this package**; kernels are implemented with mcTileLang or hand-written MACA C.

## Products

| id | arch | APs | L2 | HBM (used by the model) | BF16 | FP8 | INT8 |
|---|---|---:|---:|---:|---:|---:|---:|
| `maca.c500` | `maca.xcore1000` | 104 | 8 MiB | 1418.7 GB/s (measured copy ceiling; 1.80 TB/s nominal) | 240 TFLOPS (PCIe) | unavailable (emulated) | 480 TOPS |
| `maca.c600` | `maca.xcore1500` | 32 | 2 MiB | 1638 GB/s (nominal) | 193.35 TFLOPS | unavailable (emulated) | 386.7 TOPS |
| `maca.c600u` | `maca.xcore1600` | 28 | 2 MiB | 1183 GB/s (measured streaming ceiling; 2.048 TB/s nominal) | 219.05 TFLOPS | 438.1 TFLOPS | 438.1 TOPS |
| `maca.c600ul` | `maca.xcore1600` | 32 | 2 MiB | 1183 GB/s (C600U ceiling used as estimate) | 275.25 TFLOPS | 550.5 TFLOPS | 550.5 TOPS |

The BF16/FP8/INT8 peaks and the nominal HBM rates come from the workspace
`maca-perf-model` skill (C500/C600/C600U/C600UL table). The measured streaming
ceilings and occupancy facts come from `mcDeepGemm/docs/mega_mhc/*` and
`docs/mega_mhc_maca_c_occupancy_20261017.md`.

## Install and use

```bash
# TileFoundry and the CUDA fact machinery
python -m pip install -e /path/to/TileFoundry
python -m pip install -e /path/to/TileFoundry/targets/nvidia --no-deps
python -m pip install -e /path/to/TileFoundry/targets/maca  --no-deps

tilefoundry target list
tilefoundry target show maca.c600u
```

In an authored HIR:

```python
from tilefoundry_maca import MacaTarget

_TARGET = MacaTarget("maca.c600u")

@module(entry="s3", target=_TARGET, topologies=(...))
class S3: ...
```

then:

```bash
tilefoundry analyze model.py:S3 out.txt --compute-cost --memory --roofline
```

## Python 3.10

The tree is installable on Python 3.10 so TileFoundry can share the MACA
environment (MACA torch and mcTileLang are cp310-only). `requires-python` is
`>=3.10` in the core and target distributions. The source changes:

- 15 PEP 695 generic declarations → `TypeVar`/`Generic`; `type ShapeDim = ...`
  → `ShapeDim: TypeAlias = ...`.
- `StrEnum` and `tomllib` → `tilefoundry/_compat.py` shims (`tomli` on 3.10).
- `Tuple[is_ranked_tensor()]` (a pattern used as a type argument) →
  `tuple_annotation(...)` (`types.GenericAlias`), because 3.10's
  `typing.Tuple` rejects non-types.
- Four record fields `events: ClassVar` → `events: ClassVar[tuple[str, ...]]`:
  with `from __future__ import annotations`, 3.10's `get_type_hints` rejects
  the string form `'ClassVar'` as a type argument.

Install in the py310 MACA environment (the internal index has `isl-python`
0.1.9 and `ortools` 9.9 for cp310; both are below the py312 versions but the
`analyze` path was validated with them):

```bash
python -m pip install isl-python==0.1.9 ortools==9.9.3963 graphviz \
    pandas pytz tzdata absl-py immutabledict protobuf
python -m pip install "setuptools-scm==8.1.0"
python -m pip install -e /path/to/TileFoundry --no-deps --no-build-isolation
python -m pip install -e /path/to/TileFoundry/targets/nvidia --no-deps --no-build-isolation
python -m pip install -e /path/to/TileFoundry/targets/maca --no-deps --no-build-isolation
tilefoundry target list
```

`ortools` imports `pandas` from `cp_model`; `setuptools-scm` 10.x needs
`vcs_versioning`, so 8.1.0 is used for the editable builds. `transformers` is
only needed by `models`/orchestrator, not by `analyze`.

The workspace py310 environment already has `apache-tvm-ffi`, `Jinja2`, `tomli`,
MACA torch and `tilelang`; the three editable installs above add TileFoundry and
the two target providers.

## Provenance and caveats

Every fact carries `origin` and `conditions` in its TOML. Most numbers are
`reference`/`derived`; the following are **estimates** and must be replaced
before using a roofline as a hard gate:

- `maca.c600u`: `compute.sm_clock_hz` = 1900 MHz (mx-smi `--show-clock`) and
  `memory.gmem.latency` = 800 ns (pointer-chase, 769-822 ns at 1-4 chains/AP;
  L2-resident 308-323 ns). The C500/C600/C600UL clocks and latencies remain
  estimates (1.5 GHz / 600 ns) until measured on those products.
- `memory.shared.bandwidth` = 128 B/clock/AP is an estimate on every product.

The analysis reuses the CUDA topology vocabulary: `cta` = AP, `thread` = a
64-lane wave. The HIR's mesh must be physically placeable on the stated
resource limits; a coarse mesh (for example 128 tokens per CTA with a 1024-wide
hidden tile) can exceed `rmem`/L2 capacity and still reports a conservative L2
advisory. The core also refuses a whole-grid footprint that cannot be
restricted to one wave instead of multiplying it by the wave count; after that
fix and the measured 800 ns / 1900 MHz facts, the S3 example reports 2.11 ms
(1.58 ms with the earlier 600 ns estimate) rather than the earlier 15.7 ms. See
`mcDeepGemm/docs/experiments/20261026_c600u_mega_mhc_tilelang_s3/README.md` for
the worked example.

Adding a product needs no code: copy a `maca_*.toml` device document, set its
facts and `[compatibility] architectures`, keep the schema
`tilefoundry.cuda.device/v3`, and add it under
`tilefoundry_maca/hardware/`.
