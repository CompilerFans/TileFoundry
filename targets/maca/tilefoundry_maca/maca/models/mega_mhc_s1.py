"""Price a MACA-C kernel's S1 pass with TileFoundry's analysis.

`mega_mhc` (mcDeepGemm, `bf16_mega_mhc_maca_c.cuh`) runs its residual mix and
prenorm GEMM over hidden c-tiles.  This model keeps the kernel's traffic and
its per-CTA weight re-read, which is what decides the HBM roofline:

* each CTA owns `BM` tokens and reads its own slice of `x` / `residual`;
* `fn` has no CTA axis, so every CTA reads the whole weight set
  (`24 x 4 x H` fp32) — at BM=128 that is 672 MiB over 256 CTAs, and at BM=64
  it doubles to 1.34 GiB;
* the GEMM's arithmetic is ~45 GFLOP over the whole call, so the model carries
  the traffic and lets `--roofline` say `bound-by=memory`.

Run from the TileFoundry checkout:

    tilefoundry analyze targets/maca/examples/mega_mhc_s1.py:S1MixGemm \
        /tmp/s1.txt --compute-cost --memory --roofline

The `wsum` output exists only to keep the weight read in the dataflow; it is
not part of the kernel.
"""
from tilefoundry.dsl import *
from tilefoundry_maca import MacaTarget

T = 32768      # production tokens
H = 7168       # hidden
R = 4          # routes (hc_mult)
N = 24         # prenorm outputs (the kernel pads to 32)
BM = 128       # tokens per CTA: 128 like ct128, 64 like the ct64 arm
CTAS = T // BM

_TARGET = MacaTarget("maca.c600u")
_CTA = Topology("cta", CTAS)
_THREAD = Topology("thread", 256)


@module(entry="s1", target=_TARGET, topologies=(_CTA, _THREAD))
class S1MixGemm:
    @func
    def s1(x: Tensor[(T, H), "bf16"],
           residual: Tensor[(T, R, H), "bf16"],
           fn: ConstTensor[(N, R, H), "f32"]):
        with Mesh(("cta",), (CTAS,), ("tile",)) as cta:
            with Mesh(("thread",), (256,), ("t",)) as th:
                xt = tf.reshard(x, (T @ cta.tile, H @ th.t), "rmem")
                rt = tf.reshard(
                    residual, (T @ cta.tile, R, H @ th.t), "rmem")
                # No CTA axis on fn: the analysis charges it per CTA.
                ft = tf.reshard(fn, (N, R, H @ th.t), "rmem")
                nr = rt + tf.reshape(xt, new_shape=(T, 1, H))
                x1 = tf.reduce(nr, (1,), False, ReduceKind.SUM)
                wsum = tf.reduce(ft, (0, 1), False, ReduceKind.SUM)
                return (
                    tf.reshard(nr, (T @ cta.tile, R, H), "gmem"),
                    tf.reshard(x1, (T @ cta.tile, H), "gmem"),
                    tf.reshard(wsum, (H,), "gmem"),
                )
