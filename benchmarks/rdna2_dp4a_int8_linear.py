#!/usr/bin/env python3
"""RDNA2 INT8 linear microbenchmark for the DP4A path.

Measures ``ck.int8_linear`` on the HIP backend across the M=16 GEMV/GEMM
crossover and compares it with the eager path that RDNA2 users got before the
DP4A path existed, plus a dense FP16 linear as a bandwidth/compute anchor.

Semantics note: on gfx10 the eager path computes the same quantized-int8
contract with a float GEMM (hipBLASLt's gfx10 Tensile library is missing from
common ROCm wheels), so the eager column is the pre-DP4A working path at the
same semantic level. The DP4A correctness contract is covered by
tests/test_hip_wmma.py against exact INT8 math.

Run on an RDNA2 machine with a ROCm PyTorch:

    python benchmarks/rdna2_dp4a_int8_linear.py --reps 20
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as functional

import comfy_kitchen as ck

# GEMV sweep around the M=16 crossover, tile-boundary shapes, and realistic
# diffusion GEMMs.
SHAPES = [
    (1, 256, 256),
    (2, 256, 256),
    (4, 256, 256),
    (8, 256, 256),
    (16, 512, 256),
    (17, 256, 512),
    (32, 256, 512),
    (64, 256, 512),
    (128, 512, 256),
    (65, 67, 80),
    (128, 33, 144),
    (1024, 512, 272),
    (1024, 3072, 3072),
    (1024, 3072, 12288),
    (1024, 12288, 3072),
]


def _bench(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), min(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--json-out", default=None, help="write results as JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise SystemExit("requires a ROCm PyTorch with a visible GPU")

    props = torch.cuda.get_device_properties(0)
    print(f"device: {torch.cuda.get_device_name(0)}  arch: {props.gcnArchName}")

    rows = []
    for m, n, k in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.5
        w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.5
        wq, ws = ck.quantize_int8_rowwise(w)
        ws = ws.reshape(-1)
        bias = torch.randn(n, device="cuda", dtype=torch.float32)

        def hip_run(x=x, wq=wq, ws=ws, bias=bias):
            with ck.use_backend("hip"):
                return ck.int8_linear(x, wq, ws, bias, torch.float32)

        def eager_run(x=x, wq=wq, ws=ws, bias=bias):
            with ck.use_backend("eager"):
                return ck.int8_linear(x, wq, ws, bias, torch.float32)

        out = hip_run()
        assert out.shape == (m, n)

        hip_ms, hip_min = _bench(hip_run, args.reps)
        eager_ms, eager_min = _bench(eager_run, args.reps)

        x16 = x.half()
        w16 = w.half()
        b16 = bias.half()
        fp16_ms, fp16_min = _bench(
            lambda x16=x16, w16=w16, b16=b16: functional.linear(x16, w16, b16), args.reps
        )

        rows.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "hip_dp4a_ms": hip_ms,
                "hip_dp4a_ms_min": hip_min,
                "eager_ms": eager_ms,
                "eager_ms_min": eager_min,
                "fp16_dense_ms": fp16_ms,
                "fp16_dense_ms_min": fp16_min,
                "dp4a_vs_legacy_eager": hip_ms / eager_ms,
                "dp4a_vs_fp16": hip_ms / fp16_ms,
                "dp4a_tops": 2 * m * n * k / (hip_ms * 1e-3) / 1e12,
                "fp16_tflops": 2 * m * n * k / (fp16_ms * 1e-3) / 1e12,
            }
        )
        print(
            f"{m:5d} x {n:5d} x {k:5d} | "
            f"dp4a {hip_ms:8.3f} ms | eager {eager_ms:8.3f} ms | "
            f"dp4a/eager {hip_ms / eager_ms:5.2f}x | "
            f"fp16 {fp16_ms:8.3f} ms | dp4a/fp16 {hip_ms / fp16_ms:5.2f}x | "
            f"dp4a {2 * m * n * k / (hip_ms * 1e-3) / 1e12:6.2f} TOPS | "
            f"fp16 {2 * m * n * k / (fp16_ms * 1e-3) / 1e12:6.2f} TFLOPS"
        )

    summary = {
        "device": torch.cuda.get_device_name(0),
        "arch": props.gcnArchName,
        "reps": args.reps,
        "rows": rows,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
